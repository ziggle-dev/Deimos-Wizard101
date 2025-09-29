import asyncio
import time
import traceback
import math

from loguru import logger

import re
from src.auto_pet import auto_pet
from src.teleport_math import *
from wizwalker import XYZ, Keycode, MemoryReadError, Client, Rectangle, HookAlreadyActivated, HookNotActive
from wizwalker.file_readers.wad import Wad
from wizwalker.memory import DynamicClientObject
from wizwalker.extensions.scripting import teleport_to_friend_from_list
from src.sprinty_client import SprintyClient
from src.utils import *
from src.paths import *
from thefuzz import fuzz


class Quester():
    def __init__(self, client: Client, clients: list[Client], leader_pid: int):
        self.client = client
        self.clients = clients
        self.leader_pid = leader_pid
        self.current_leader_client = client
        self.current_leader_pid = leader_pid
        self.d_location = None

    async def read_quest_txt(self, client: Client) -> str:
        try:
            quest_name = await get_window_from_path(client.root_window, quest_name_path)
            quest = await quest_name.maybe_text()
        except:
            quest = ""
        return quest

    async def read_spiral_door_title(self, client: Client) -> str:
        try:
            title_text_path = await get_window_from_path(client.root_window, spiral_door_title_path)
            title = await title_text_path.maybe_text()
        except:
            title = ""

        return title

    async def read_popup(self, p: Client) -> str:
        try:
            popup_text_path = await get_window_from_path(p.root_window, popup_msgtext_path)
            txtmsg = await popup_text_path.maybe_text()
        except:
            txtmsg = ""
        return txtmsg

    async def detected_interact_from_popup(self, p: Client) -> bool:
        txtmsg = await self.read_popup(p)

        msg = txtmsg.lower()
        if not 'to talk' in msg and not 'to open' in msg and not 'to collect' in msg:
            return True
        else:
            return False

    async def is_position_safe(self, position: XYZ, safe_distance: float = 1000) -> bool:
        sp = SprintyClient(self.client)

        for mob in await sp.get_mobs():
            mob_pos = await mob.location()
            if math.dist(mob_pos, position) < safe_distance:
                return False

        return True

    async def get_zone_chunks(self) -> list[XYZ]:
        wad = await self.load_wad(await self.client.zone_name())
        nav_data = await wad.get_file("zone.nav")
        vertices, _ = parse_nav_data(nav_data)

        full = calc_chunks(vertices)
        return full

    async def get_collect_quest_object_name(self) -> str:
        # '<center>Collect Cog in Triton Avenue (0 of 3)</center>'
        # -> 'Cog'
        s = await self.read_quest_txt(self.client)
        res = re.findall(r"\w+\s+(.*)\s+in.*", s)
        if len(res) == 0:
            return ''
        return res[0].strip()

    # TODO: Does this need a client?
    async def get_quest_zone_name(self, c: Client) -> str:
        # <center>Collect Cog in Triton Avenue (0 of 3)</center>
        # Collect Cog in Triton Avenue (0 of 3)
        # -> 'Triton Avenue'

        query = await self.read_quest_txt(c)

        stopwords = ['<center>','</center>']
        querywords = query.split()
        resultwords  = [word for word in querywords if word.lower() not in stopwords]
        s = ' '.join(resultwords)

        res = re.findall(r"\s+in\s+([^\(]*)", s)
        if len(res) == 0:
            return ''
        return res[0].strip()

    async def get_truncated_quest_objectives(self, p: Client) -> str:
        quest_objective = await get_quest_name(p)
        if '(' in quest_objective:
            quest_objective = quest_objective.split('(', 1)
            quest_objective = quest_objective[0]

        return quest_objective

    async def load_wad(self, path: str):
        return Wad.from_game_data(path.replace("/", "-"))

    async def followers_in_correct_zone(self) -> bool:
        zone = await self.current_leader_client.zone_name()
        for c in self.clients:
            if await c.zone_name() != zone:
                return False
        return True

    async def determine_solo_zone(self) -> bool:
        sprinter = SprintyClient(self.current_leader_client)
        entities = await sprinter.get_base_entity_list()

        player_count = 0
        for entity in entities:
            entity_name = await entity.object_name()
            if entity_name == 'Player Object':
                player_count += 1
                if player_count > 1:
                    return False
        return True

    async def get_follower_clients(self) -> list[Client]:
        follower_clients = []
        for c in self.clients:
            if c.process_id != self.current_leader_client.process_id:
                follower_clients.append(c)

        return follower_clients

    # get a list of clients that are questing alongside the leader, including the leader (not boosting)
    async def get_questing_clients(self) -> list[Client]:
        questing_clients = [self.current_leader_client]
        quest_objective_leader = await self.get_truncated_quest_objectives(self.current_leader_client)

        for c in await self.get_follower_clients():
            quest_objective_follower = await self.get_truncated_quest_objectives(c)
            if quest_objective_follower == quest_objective_leader:
                questing_clients.append(c)

        return questing_clients

    # get a dict of client quests for clients that are on the same quest as the leader, including the leader
    async def get_client_quests(self, questing_clients: list[Client]) -> dict[Client, str]:
        client_quests = {}

        for c in questing_clients:
            quest_objective = await self.get_truncated_quest_objectives(c)
            client_quests[c] = quest_objective

        return client_quests

    async def zone_recorrect_hub(self):
        if await self.followers_in_correct_zone():
            return
        for p in self.clients:
            await p.send_key(Keycode.END)
            await p.send_key(Keycode.END)
            await asyncio.sleep(3)
            # use is_loading instead of wait_for_change, as one client could already be in the hub
            while await p.is_loading():
                await asyncio.sleep(0.1)

        await asyncio.sleep(2)

        # Track entities in new zone
        from src.quest_db import get_quest_db
        db = get_quest_db()
        await db.track_zone_entities(self.current_leader_client)

    async def friend_teleport(self, maybe_solo_zone: bool):
        clients_in_solo_zone = []
        solo_zone = None
        leader_in_solo_zone = False
        was_loading = False

        for c in self.clients:
            if c.process_id != self.current_leader_pid:
                c_zone = await c.zone_name()
                leader_zone = await self.current_leader_client.zone_name()
                if c_zone != leader_zone or maybe_solo_zone:
                    if await is_free(c) and not c.entity_detect_combat_status:
                        # teleport to leader
                        await c.send_key(Keycode.F, 0.1)
                        ##############################################
                        async with c.mouse_handler:
                            await teleport_to_friend_from_list(c, name=self.current_leader_client.wizard_name)  # icon_list=1, icon_index=self.current_leader_client.questing_friend_teleport_icon)
                        if c_zone != leader_zone:
                            async with c.mouse_handler:
                                try:
                                    await safe_wait_for_zone_change(c, handle_hooks_if_needed=False)
                                except FriendBusyOrInstanceClosed:
                                    leader_in_solo_zone = True
                                    solo_zone = await self.current_leader_client.zone_name()

                                    # if leader is in solo zone, others may be too - meaning the user is likely trying to quest multiple clients at the same time.  Keep track of these questing clients
                                    for p in self.clients:
                                        if await p.zone_name() == await self.current_leader_client.zone_name():
                                            clients_in_solo_zone.append(p)

                                    break
                                except LoadingScreenNotFound:
                                    print(traceback.print_exc())
                                    while True:
                                        await asyncio.sleep(1.0)

                                was_loading = True

                        else:
                            await asyncio.sleep(6.0)
                            if await is_visible_by_path(c, friend_is_busy_and_dungeon_reset_path):
                                async with c.mouse_handler:
                                    while await is_visible_by_path(c, friend_is_busy_and_dungeon_reset_path):
                                        leader_in_solo_zone = True
                                        await click_window_by_path(c, friend_is_busy_and_dungeon_reset_path)

                                solo_zone = await self.current_leader_client.zone_name()

                                # if leader is in solo zone, others may be too - meaning the user is likely trying to quest multiple clients at the same time.  Keep track of these questing clients
                                for p in self.clients:
                                    if await p.zone_name() == await self.current_leader_client.zone_name():
                                        clients_in_solo_zone.append(p)

                                break

        if not leader_in_solo_zone:
            maybe_solo_zone = False

        # leaving a loading screen is not equivalent to being ready to move - give clients time to truly load into a zone
        if was_loading:
            await asyncio.sleep(3.0)

        return clients_in_solo_zone, solo_zone

    # handle zone recorrection in follow leader mode using friend TP.  Also quest all questing clients individually when they are in a solo zone

    # maybe_solo_zone - tells the function that we think we may be in a solo zone, and that it should check and account for that even if we are technically in the same zone as the leader
    async def zone_recorrect_friend_tp(self, maybe_solo_zone: bool, gear_switching_in_solo_zones=False):
        # for solo zone questing support across multiple clients
        async def solo_zone_questing_loop(clients_in_solo: List[Client], zone: str):
            async def solo_zone_questing(solo_cl: Client):
                questing = Quester(solo_cl, self.clients, None)
                while solo_cl.questing_status and await solo_cl.zone_name() == zone:
                    await asyncio.sleep(1.0)

                    if solo_cl in self.clients and solo_cl.questing_status:
                        await questing.auto_quest_solo(auto_pet_disabled=True)

            await asyncio.gather(*[solo_zone_questing(cl) for cl in clients_in_solo])

        if len(self.clients) > 1:
            # if clients are not in same zone as leader, teleport there, or quest leader individually
            while not await self.followers_in_correct_zone() or maybe_solo_zone:
                clients_in_solo_zone, solo_zone = await self.friend_teleport(maybe_solo_zone)
                initial_clients_in_solo_zone = clients_in_solo_zone.copy()

                # if client(s) in solo zone, switch to non-leader auto questing and quest each valid client on their own until their zone changes
                # deck_switching = True
                if len(clients_in_solo_zone) > 0:
                    for solo_client in clients_in_solo_zone:
                        logger.debug('Client ' + solo_client.title + ' is in solo zone - questing alone')

                        solo_client.in_solo_zone = True

                    if gear_switching_in_solo_zones:
                        logger.debug('Switching to second equipment set on all clients.')
                        await asyncio.gather(*[change_equipment_set(c, 1,) for c in clients_in_solo_zone])

                    # loop until we have confirmed that we are no longer in a solo zone
                    while len(clients_in_solo_zone) > 0 and solo_zone is not None:
                        solo_zone_task = asyncio.create_task(solo_zone_questing_loop(clients_in_solo=clients_in_solo_zone, zone=solo_zone))
                        await asyncio.wait([solo_zone_task])

                        logger.debug('Clients may have left the solo zone - attempting to teleport to leader.')
                        clients_in_solo_zone, solo_zone = await self.friend_teleport(maybe_solo_zone=True)

                    # we have confirmed that we are out of the solo zone(s) - carry on questing
                    maybe_solo_zone = await self.determine_solo_zone()
                    if maybe_solo_zone:
                        logger.debug('Some clients appear to still be in the solo zone.')
                    else:
                        logger.debug('Clients all appear to have left the solo zone.')

                    for solo in initial_clients_in_solo_zone:
                        solo.in_solo_zone = False

                    if gear_switching_in_solo_zones:
                        logger.debug('Switching back to first equipment set on all clients.')
                        await asyncio.gather(*[change_equipment_set(c, 0,) for c in initial_clients_in_solo_zone])
                else:
                    maybe_solo_zone = await self.determine_solo_zone()

        return maybe_solo_zone

    async def heal_and_handle_potions(self):
        await asyncio.gather(*[self.collect_wisps(p) for p in self.clients])
        await asyncio.gather(*[self.guarantee_use_potion(p) for p in self.clients])

        any_client_needs_potions = False
        for p in self.clients:
            # If we have less than 1 potion left, send all clients to get potions (even if some don't need it).  Only do this once per questing loop
            if await p.stats.potion_charge() < 1.0 and await p.stats.reference_level() >= 6:
                any_client_needs_potions = True

        if any_client_needs_potions:
            # Guaranteed teleport mark placement - marks until mana is lower than starting point
            for p in self.clients:
                if await p.zone_name() != 'WizardCity/WC_Hub':
                    original_mana = await p.stats.current_mana()
                    while await p.stats.current_mana() >= original_mana:
                        logger.debug(f'Client {p.title} - Marking Location')
                        await p.send_key(Keycode.PAGE_DOWN, 0.1)
                        await asyncio.sleep(.75)

            await asyncio.gather(*[refill_potions(c, mark=False, recall=False, original_zone=await c.zone_name()) for c in self.clients])
            # return all clients to original location, and if it fails, send all to commons
            await asyncio.gather(*[self.gather_clients_from_potion_buy(c) for c in self.clients])

    async def collect_wisps(self, p: Client):
        if await is_free(p):
            if await is_potion_needed(p) and await p.stats.current_mana() > 1 and await p.stats.current_hitpoints() > 1:
                await collect_wisps(p)

    async def guarantee_use_potion(self, p: Client):
        if await is_free(p):
            if await is_potion_needed(p, minimum_mana=16):
                original_potion_count = await p.stats.potion_charge()
                while await p.stats.potion_charge() == original_potion_count and original_potion_count >= 1:
                    logger.debug(f'Client {p.title} - Using potion')
                    await click_window_by_path(p, potion_usage_path, True)
                    await asyncio.sleep(.6)

    async def gather_clients_from_potion_buy(self, p: Client):
        while True:
            try:
                await p.send_key(Keycode.PAGE_UP)
                logger.debug('Waiting for zone change on all clients.')
                await safe_wait_for_zone_change(p, name='WizardCity/WC_Hub', handle_hooks_if_needed=True)
                break
            except LoadingScreenNotFound:
                pass
            # something went wrong when placing the mark or recalling / our teleport mark was in a closed off area / instance
            # send all to the commons and let auto quest handle the rest
            except FriendBusyOrInstanceClosed:
                for c in self.clients:
                    while await c.zone_name() != 'WizardCity/WC_Hub':
                        await navigate_to_ravenwood(c)
                        await navigate_to_commons_from_ravenwood(c)
                        await asyncio.sleep(.5)

                break

    # if followers in different zone, try X presses (for X zone changes that require delayed presses between clients)
    async def X_press_zone_recorrect(self):
        for c in self.clients:
            if c.process_id != self.current_leader_pid:
                await c.send_key(Keycode.X, 0.1)
                await asyncio.sleep(2.5)

        await asyncio.sleep(2)
        for c in self.clients:
            while await c.is_loading():
                await asyncio.sleep(0.1)

    # async def enter_dungeon(self):
        # Handles entering dungeons
        # await asyncio.gather(*[p.send_key(Keycode.X, 0.1) for p in self.clients])

        # await asyncio.sleep(1.5)

        # for c in self.clients:
        #     if await is_visible_by_path(c, dungeon_warning_path):
        #         await c.send_key(Keycode.ENTER, 0.1)

        # await asyncio.gather(*[c.wait_for_zone_change() for c in self.clients])

    async def find_quest_zone_area_name(self, client: Client, door_locations: list) -> Optional[str]:
        location = await self.get_quest_zone_name(client)
        location = location.lower()
        parts_of_string = location.split(" ")
        for piece in parts_of_string:
            for d_location in door_locations:
                if piece in d_location:
                    self.d_location = d_location
                    return d_location

    async def new_world_doors(self, client: Client) -> bool:
        if "Streamportal" in await self.read_spiral_door_title(client):
            location = await self.find_quest_zone_area_name(client, streamportal_locations)
            await new_portals_cycle(client, location)
            return True

        elif "Nanavator" in await self.read_spiral_door_title(client):
            location = await self.find_quest_zone_area_name(client, nanavator_locations)
            await new_portals_cycle(client, location)
            return True

        return False

    async def handle_spiral_navigation(self):
        if await self.new_world_doors(self.current_leader_client):
            for c in self.clients:
                if c.process_id != self.current_leader_pid:
                    if await is_visible_by_path(c, spiral_door_teleport_path):
                        await new_portals_cycle(c, self.d_location)
        else:
            # Handles spiral door navigation
            await spiral_door_with_quest(self.current_leader_client)

            await asyncio.sleep(1)
            leader_world = (await self.current_leader_client.zone_name()).split('/', 1)[0]

            # Follower clients use separate spiral door navigation than leader (since they may not have the same quest)
            for c in self.clients:
                if c.process_id != self.current_leader_pid:
                    if await is_visible_by_path(c, spiral_door_teleport_path):
                        await go_to_new_world(c, leader_world)

    async def auto_tfc_friend_all_wizards(self):
        for code_generator in self.clients:
            for code_redeemer in self.clients:
                if code_generator.process_id != code_redeemer.process_id:
                    friend_already_in_list = await check_for_friend_in_list(code_generator, code_redeemer.wizard_name)

                    if not friend_already_in_list:
                        tfc = await generate_tfc(code_generator)
                        await accept_tfc(code_redeemer, tfc)

    # This is horribly inconsistent due to wizwalker reading incorrect values
    async def auto_friend_all_wizards(self):
        # await self.current_leader_client.send_key(Keycode.END, 0.1)
        # await asyncio.sleep(4)
        # while await self.current_leader_client.is_loading():
        #    await asyncio.sleep(.1)

        # move leader forward so they are not overlapped with other arriving clients
        await self.current_leader_client.send_key(Keycode.W, .5)

        await asyncio.sleep(2)
        await asyncio.gather(*[navigate_to_ravenwood(p) for p in self.clients])
        # await toZone(self.clients, 'WizardCity/WC_Ravenwood')  # await self.current_leader_client.zone_name())

        # send all to a common realm (Centaur)
        for c in self.clients:
            while not await is_visible_by_path(c, check_spellbook_open_path):
                await c.send_key(Keycode.ESC, 0.1)

            async with c.mouse_handler:
                for i in range(3):
                    await c.mouse_handler.click_window_with_name('RealmsButton')
                    await asyncio.sleep(.1)

                for i in range(3):
                    await c.mouse_handler.click_window_with_name('btnRealm' + str(6))
                    await asyncio.sleep(.1)

                for i in range(3):
                    if not await c.is_loading():
                        try:
                            await c.mouse_handler.click_window_with_name('btnGoToRealm')
                        except ValueError:
                            await asyncio.sleep(.1)
                        await asyncio.sleep(.5)

                await asyncio.sleep(2.0)
                while await c.is_loading():
                    await asyncio.sleep(.1)

                if await is_visible_by_path(c, close_spellbook_path):
                    await click_window_by_path(c, close_spellbook_path)

        for requester in self.clients:
            requester_original_position = await requester.body.position()
            # teleport to a wacky spot in Ravenwood so that we dont accidentally click the wrong player
            # yaw moves our camera really close (because we're colliding with a wall), reducing the chance of mis-clicking
            await requester.teleport(XYZ(x=-1884.0, y=-2328.0, z=0.0), yaw=0.7693982720375061)
            await asyncio.sleep(.3)
            await requester.send_key(Keycode.W, 0.1)
            for acceptor in self.clients:
                if requester.process_id != acceptor.process_id:
                    friend_already_in_list = await check_for_friend_in_list(requester, acceptor.wizard_name)

                    if not friend_already_in_list:
                        original_acceptor_location = await acceptor.body.position()
                        # teleport the clients that are friending each other to the same location
                        await acceptor.teleport(await requester.body.position(), yaw=await requester.body.yaw())
                        await asyncio.sleep(10)

                        # Click a few times initially and pray it works, as wiz sometimes lies about the add friend window being visible when it isn't

                        async with requester.mouse_handler:
                            rect: Rectangle = requester.window_rectangle
                            width = (rect.x2 - rect.x1)
                            height = (rect.y2 - rect.y1)
                            # Width and Height are always off by a set number - unsure if this interferes with clicking
                            # width = abs(width + 16)
                            # height = height - 39
                            width = abs(width)
                            center_x = width / 2
                            center_y = height / 2

                            for i in range(5):
                                await requester.mouse_handler.click(x=int(center_x), y=int(center_y), sleep_duration=0.3)
                                await asyncio.sleep(.2)

                            # friend_title = await get_friend_popup_wizard_name(requester)

                            # continually click until the correct friend popup window appears
                            # this code may never run, as reading the friend popup window is inaccurate
                            # while friend_title != acceptor.wizard_name:
                            #     await requester.send_key(Keycode.D, 0.3)
                            #     await requester.mouse_handler.click(x=int(center_x), y=int(center_y), sleep_duration=0.3)
                            #     friend_title = await get_friend_popup_wizard_name(requester)

                            # Click add friend
                            for i in range(2):
                                if await is_visible_by_path(requester, add_remove_friend_path):
                                    await click_window_by_path(requester, add_remove_friend_path)

                            # Confirm sending the request
                            for i in range(2):
                                if await is_visible_by_path(requester, confirm_send_friend_request):
                                    await asyncio.sleep(.2)
                                    await click_window_by_path(requester, confirm_send_friend_request)

                            # Wait for accept friend popup to appear
                            for i in range(2):
                                while not await is_visible_by_path(acceptor, confirm_accept_friend_request):
                                    await asyncio.sleep(.1)

                            async with acceptor.mouse_handler:
                                # Accept friend request
                                for i in range(2):
                                    if await is_visible_by_path(acceptor, confirm_accept_friend_request):
                                        await asyncio.sleep(.4)
                                        await click_window_by_path(acceptor, confirm_accept_friend_request)
                            # Close friend window on requestor
                            # This fails consistently, even when the friends list is actually open.  Detecting whether the friends list is open is also horrifically inconsistent so just brute force it
                                for i in range(5):
                                    try:
                                        await click_window_by_path(requester, close_real_friend_list_button_path)
                                        await asyncio.sleep(.1)
                                    except ValueError:
                                        await asyncio.sleep(.1)

                        await acceptor.teleport(original_acceptor_location)
                        await acceptor.send_key(Keycode.W, 0.1)
                        await asyncio.sleep(1)

            await requester.teleport(requester_original_position)

    async def num_clients_in_same_area(self) -> int:
        sprinter = SprintyClient(self.current_leader_client)
        entities = await sprinter.get_base_entity_list()

        player_count = 0
        for entity in entities:
            entity_gid = await entity.global_id_full()
            for c in self.clients:
                client_gid = await c.client_object.global_id_full()
                if client_gid == entity_gid:
                    player_count += 1
                    break
            if player_count == len(self.clients):
                break

        return player_count

    async def determine_new_leader_and_followers(self, client_quests: dict, questing_clients: list[Client],
                                                    follower_clients: list[Client]) -> tuple[list[Client], dict[Client, str]]:
            original_length = len(client_quests)
            if len(client_quests) > 0:
                for c in questing_clients:
                    if c in client_quests:
                        if await self.get_truncated_quest_objectives(c) != client_quests.get(c):
                            client_quests.pop(c)

                # if all clients have moved on from their previous quest
                if len(client_quests) == 0:
                    # pass leader back to original leader client
                    if self.current_leader_client.title != self.client.title:
                        logger.debug('Clients caught up - resetting leader to client ' + self.client.title)
                        self.current_leader_client = self.client
                        self.current_leader_pid = self.client.process_id
                        follower_clients = await self.get_follower_clients()

                    client_quests = await self.get_client_quests(questing_clients=questing_clients)

                elif len(client_quests) < original_length:
                    # pass leader to next client in dict
                    logger.debug('client(s) fell behind - new leader ' + list(client_quests)[0].title + ' assigned')
                    self.current_leader_client = list(client_quests)[0]
                    self.current_leader_pid = self.current_leader_client.process_id
                    follower_clients = await self.get_follower_clients()
            else:
                client_quests = await self.get_client_quests(questing_clients=questing_clients)

            return follower_clients, client_quests

    async def correct_dungeon_desync(self, follower_clients):
        # attempt to correct the desync by simply teleporting all clients to the leader
        async def teleport_to_friend_from_list_wizard_leader_name_mouse_handler(client):
            async with client.mouse_handler:
                await teleport_to_friend_from_list(client, name=self.current_leader_client.wizard_name)

        await asyncio.gather(*[teleport_to_friend_from_list_wizard_leader_name_mouse_handler(c) for c in follower_clients])

        await asyncio.sleep(1.5)

        # this may fail - some zones cannot be teleported to even if they have public sigils - detect whether this zone is locked off to teleports
        teleport_banned_zone = False
        for c in self.clients:
                while await is_visible_by_path(c, friend_is_busy_and_dungeon_reset_path):
                    async with c.mouse_handler:
                        await click_window_by_path(c, friend_is_busy_and_dungeon_reset_path)
                    teleport_banned_zone = True

        await asyncio.sleep(.5)

        # if we cannot teleport, send all to hub, then teleport
        # teleporting can never fail in the hub (due to a locked zone at least), so this is a surefire way get all clients in the same realm / area
        if teleport_banned_zone:
            logger.debug('Friend teleport correction for dungeon desync failed - sending all clients to hub and retrying teleport')
            await self.zone_recorrect_hub()
            await asyncio.gather(*[teleport_to_friend_from_list_wizard_leader_name_mouse_handler(c) for c in follower_clients])

            await asyncio.sleep(5.0)
            for c in self.clients:
                while await c.is_loading():
                    await asyncio.sleep(.1)
        else:
            await asyncio.sleep(3.0)
            for c in self.clients:
                while await c.is_loading():
                    await asyncio.sleep(.1)


    async def auto_collect_rewrite(self, client: Client):
        quest_item_list = await self.get_collect_quest_object_name()

        entities_to_skip = ['Basic Positional', 'WispHealth', 'WispMana', 'KT_WispHealth', 'KT_WispMana', 'WispGold', 'DuelCircle', 'Player Object', 'SkeletonKeySigilArt', 'Basic Ambient', 'TeleportPad']

        chunk_cords = await self.get_zone_chunks()  # list of cords that load in chunk
        for points in chunk_cords:  # loops through the points
            points = XYZ(points.x, points.y, points.z - 550)  # sets cord to underground to avoid pull / detection
            await client.teleport(points)  # teleports under the area

            entities = await self.client.get_base_entity_list()  # gets the entity list of the map
            for e in entities:
                try:
                    object_template = await e.object_template()  # gets entity template
                    display_name_code = await object_template.display_name()  # gets display name code
                    display_name = await self.client.cache_handler.get_langcode_name(display_name_code)  # uses display name code to get display name text

                    match = fuzz.token_sort_ratio(display_name.lower(), quest_item_list.lower())  # thefuzz check if display name matches quest item.
                    print(display_name + ' : ' + str(match))

                    if match > 80:  # if strings match greater than 80 it means that it's most likely the item
                        while not await is_free(self.client) or self.client.entity_detect_combat_status:
                            await asyncio.sleep(.1)

                        if await self.collect_entity(e):  # grabs enity
                            return
                except ValueError:
                    pass
                except MemoryReadError:
                    pass
                except AttributeError:
                    pass

            # this is backup if displayname doesn't work
            for e in entities:
                try:
                    temp = await e.object_template()
                    e_name = str(await temp.object_name())  # entity name
                    if e_name not in entities_to_skip:  # helps speed things up
                        name_list = e_name.split('_')
                        if len(name_list) == 1:
                            name_list = name_list[0].split('-')

                        edited_name = ''.join(name_list[1:])
                        # do stripping symbols stuff with edited_name
                        edit_name2 = ''.join([i for i in edited_name if not i.isdigit()])
                        edit_name3 = str(edit_name2).replace("_", "")

                        match = fuzz.ratio(edit_name3.lower(), quest_item_list.lower())
                        if match > 50:
                            while not await is_free(self.client) or self.client.entity_detect_combat_status:
                                await asyncio.sleep(.1)

                            if await self.collect_entity(e):  # grabs enity
                                return
                except:
                    pass

    async def collect_entity(self, e: DynamicClientObject) -> bool:
        xyz = await e.location()  # gets entities xyz
        for _ in range(2):  # try's to collect item twice if not safe
            can_Teleport = await self.is_position_safe(xyz)  # checks if safe to collect
            if can_Teleport:
                safe_location = await self.client.body.position()
                try:
                    await navmap_tp(self.client, xyz)  # teleports to the xyz
                except:
                    print(traceback.format_exc())

                await asyncio.sleep(.2)  # waits 2 secs for the UI to load

                if await is_visible_by_path(self.client, path=npc_range_path):  # checks if there is an UI
                    for i in range(5):
                        await asyncio.gather(*[p.send_key(Keycode.X, .1) for p in self.clients])  # trys to collect by spamming x

                    while not await is_free(self.client) or self.client.entity_detect_combat_status:
                        await asyncio.sleep(.1)

                    # return client to their previous safe location before grabbing the entity
                    await self.client.teleport(safe_location)
                    return True
        return False

    async def dungeon_recall(self, p: Client) -> Optional[bool]:
        original_zone = await p.zone_name()
        dungeon_recalled = await click_window_until_closed(p, dungeon_recall_path)

        if dungeon_recalled:
            while original_zone == await p.zone_name():
                await asyncio.sleep(1.0)
                dungeon_full = await click_window_until_closed(p, friend_is_busy_and_dungeon_reset_path)
                if dungeon_full:
                    await asyncio.sleep(1.0)
                    return False

            return True

    async def open_character_screen(self, p: Client):
        while not await is_visible_by_path(p, close_spellbook_path):
            await p.send_key(Keycode.C, 0.1)
            await asyncio.sleep(.3)

    async def close_character_screen(self, p: Client):
        while await is_visible_by_path(p, close_spellbook_path):
            await p.send_key(Keycode.C, 0.1)
            await asyncio.sleep(.3)

    async def handle_dungeon_recall(self, follower_clients):
        # don't recall on secondary clients unless we've already successfully recalled on the primary
        dungeon_recalled = await self.dungeon_recall(self.current_leader_client)
        if dungeon_recalled:
            await asyncio.gather(*[self.dungeon_recall(p) for p in follower_clients])

    async def auto_pet_questing(self, questing_clients: list[Client], ignore_pet_level_up, play_dance_game):
        auto_pet_on = False
        for p in self.clients:
            if p.auto_pet_status:
                auto_pet_on = True
                break

        # train pet on all clients if any questing client levels up
        if auto_pet_on:
            any_client_leveled_up = False
            for c in questing_clients:
                char_level = await c.stats.reference_level()
                if char_level > c.character_level:
                    any_client_leveled_up = True
                    break

            if any_client_leveled_up:
                logger.debug('One or more questing clients leveled up - training pets on all questing clients.')
                await asyncio.gather(*[auto_pet(c, ignore_pet_level_up, play_dance_game, questing=True) for c in self.clients])

    async def handle_zone_correction(self, maybe_solo_zone: bool, questing_friend_tp: bool, gear_switching_in_solo_zones: bool):
        # If zone changed, try to determine if we are in a solo zone
        if len(self.clients) > 1:
            maybe_solo_zone = await self.determine_solo_zone()

        # if followers in wrong zone, first attempt to click X - this may send them into the next zone if they are near an interactible door
        # don't do this when friend tp is active (as x press correction for some reason appears to be inconsistent)
        if not await self.followers_in_correct_zone() or maybe_solo_zone:
            logger.debug('Clients may be in wrong zone - attempting to correct with X press')
            await self.X_press_zone_recorrect()

            # paths to exit out of when they pop up near the end of the loop
            end_of_loop_paths = (exit_recipe_shop_path, exit_equipment_shop_path, cancel_multiple_quest_menu_path, cancel_spell_vendor, exit_snack_shop_path, exit_reagent_shop_path, exit_tc_vendor, exit_minigame_sigil, exit_wysteria_tournament, exit_dungeon_path, exit_zafaria_class_picture_button, exit_pet_leveled_up_button_path, avalon_badge_exit_button_path, potion_exit_path)
            await asyncio.gather(*[exit_menus(c, end_of_loop_paths) for c in self.clients])

        if not await self.followers_in_correct_zone() or maybe_solo_zone:
            if not questing_friend_tp:
                # if we still aren't in correct zone, send all to hub and retry
                await self.zone_recorrect_hub()
            else:
                # if still in the wrong zone, try friend teleport
                maybe_solo_zone = await self.zone_recorrect_friend_tp(maybe_solo_zone, gear_switching_in_solo_zones)

                await asyncio.sleep(2.0)

    async def handle_dungeon_entry(self, questing_friend_tp: bool, follower_clients: list[Client]):
        # await self.enter_dungeon()
        await asyncio.gather(*[c.wait_for_zone_change() for c in self.clients])

        await asyncio.sleep(1.0)

        # Track entities in the new dungeon zone
        from src.quest_db import get_quest_db
        db = get_quest_db()
        await db.track_zone_entities(self.current_leader_client)

        # check for instance desync, and attempt to fix if it has happened
        if questing_friend_tp:
            num_visible_clients = await self.num_clients_in_same_area()

            if num_visible_clients == len(self.clients):
                pass

            elif num_visible_clients > 1:
                # teleport all, this clearly isn't a solo zone
                logger.debug('One or more clients was separated from the group - teleporting all to leader')
                await self.correct_dungeon_desync(follower_clients)

    async def handle_npc_talking_quests(self, talking_client: Client, present_clients: list[Client]):
        # wait for initial dialogue to appear
        while await is_free_leader_questing(talking_client):
            await asyncio.sleep(.1)

        start_time = time.time()
        while time.time() < start_time + 3.0:
            # we most likely entered another dialogue - wait for it to end, then reset the timer
            if not await is_free_leader_questing(talking_client):
                logger.info('Detected dialogue - waiting for it to end.')
                while not await is_free_leader_questing(talking_client):
                    await asyncio.sleep(.1)

                after_talking_paths = (exit_zafaria_class_picture_button, exit_pet_leveled_up_button_path, avalon_badge_exit_button_path)
                await asyncio.gather(*[exit_menus(c, after_talking_paths) for c in present_clients])
                await asyncio.sleep(.4)

                # loop until we can make it 5 seconds without finding another dialogue
                start_time = time.time()

            await asyncio.sleep(.1)

    async def teleport_to_quest(self, hitting_client: str, follower_clients: list[Client]):
        await asyncio.gather(*[self.leader_wait_for_free(p) for p in self.clients])

        if await is_free_leader_questing(self.current_leader_client):
            leader_client_objective_xyz = await self.current_leader_client.quest_position.position()
            leader_objective = await self.get_truncated_quest_objectives(self.current_leader_client)

            # complex teleport logic for defeat quests to prevent mob battle separation
            if 'defeat' in leader_objective.lower():
                # if the hitting client is the leader client, they would teleport first, forcing them into battle first
                # we use a proxy leader client instead during battle teleports so that in the case that the hitting client is the leader, we can still teleport them last
                ignore_hitter = True
                proxy_leader_is_questing = True
                if hitting_client is not None:
                    # this is solely for cases where the user made a stupid config file and made their leader client the same as their hitter client
                    if hitting_client in self.current_leader_client.title:
                        ignore_hitter = False
                        followup_teleport_clients = follower_clients.copy()
                        proxy_leader_client = None
                        for c in follower_clients:
                            # prefer other concurrent questers as the proxy leader (in case we run into a solo zone)
                            if proxy_leader_client is None:
                                if await self.get_truncated_quest_objectives(c) == await self.get_truncated_quest_objectives(self.current_leader_client):
                                    proxy_leader_client = c
                                    break

                        # none of our follower clients are on the same quest (user has a really strange setup, this should never happen in practice)
                        # resort to letting a non-questing client be the proxy leader
                        if proxy_leader_client is None:
                            proxy_leader_is_questing = False
                            proxy_leader_client = follower_clients[0]

                        followup_teleport_clients.remove(proxy_leader_client)
                        followup_teleport_clients.append(self.current_leader_client)

                # user did not set a hitter client or the hitter client is not the current leader
                # in this case, change nothing - leader remains leader and followers remain followers
                if ignore_hitter:
                    proxy_leader_client = self.current_leader_client
                    followup_teleport_clients = follower_clients

                logger.debug('Leader ' + self.current_leader_client.title + ' on defeat quest - staggering teleports')

                location_before_sendback = await proxy_leader_client.body.position()
                zone_before_teleport = await proxy_leader_client.zone_name()
                await proxy_leader_client.teleport(leader_client_objective_xyz)
                await asyncio.sleep(1.0)

                # we collided and were sent back - we likely aren't in the right zone for our defeat quest
                distance = calc_Distance(location_before_sendback, await proxy_leader_client.body.position())

                # leader client collided and got sent back
                if distance < 20:
                    logger.debug('client ' + proxy_leader_client.title + ' collided on initial teleport')
                    await navmap_tp(client=proxy_leader_client, xyz=leader_client_objective_xyz, leader_client=self.current_leader_client)

                await asyncio.sleep(1.0)
                while await proxy_leader_client.is_loading():
                    await asyncio.sleep(.1)

                # leader_current_zone = await self.current_leader_client.zone_name()
                detected_dungeon = await self.detected_interact_from_popup(proxy_leader_client)

                # changed zones or we are in front of an interactible
                if await proxy_leader_client.zone_name() != zone_before_teleport or detected_dungeon:
                    logger.debug('leader zone changed or interactible reached - syncing all clients')
                    try:
                        await asyncio.gather(*[navmap_tp(client=c, xyz=leader_client_objective_xyz, leader_client=self.current_leader_client) for c in followup_teleport_clients])
                    except:
                        print(traceback.print_exc())

                # objective changed, let questing loop cycle and assign a new leader in case some got left behind
                # ignore this logic if the proxy leader is not a questing client - there is no way to determine if objective changed in this case and users should avoid this setup
                elif proxy_leader_is_questing and leader_objective != await self.get_truncated_quest_objectives(proxy_leader_client):
                    pass

                # leader is likely waiting for combat in the correct zone
                else:
                    logger.debug('leader waiting for combat')
                    await asyncio.sleep(1)
                    sprinter = SprintyClient(proxy_leader_client)
                    while not proxy_leader_client.entity_detect_combat_status:
                        try:
                            await sprinter.tp_to_closest_mob()
                        # wizwalker throws should update bool even with wait_on_inuse on
                        except ValueError:
                            await asyncio.sleep(1.0)

                        await asyncio.sleep(5.0)

                    while proxy_leader_client.entity_detect_combat_status:
                        await asyncio.sleep(.1)

            # if we aren't doing a mob / boss fight, we have no need to stagger teleports
            # furthermore staggered teleports can break certain quests in dungeons for certain clients
            else:
                await asyncio.gather(*[navmap_tp(p, leader_client_objective_xyz, leader_client=self.current_leader_client) for p in self.clients])

    async def handle_normal_quests(self, follower_clients: list[Client], questing_friend_tp: bool):
        # Handles chest reroll menu, will always cancel
        await asyncio.gather(*[safe_click_window(c, cancel_chest_roll_path) for c in self.clients])
        # confirm exit dungeon early button
        await asyncio.gather(*[safe_click_window(c, exit_dungeon_path) for c in self.clients])

        await asyncio.gather(*[self.leader_wait_for_free(p) for p in self.clients])

        if await is_free_leader_questing(self.current_leader_client):
            for c in self.clients:
                while await c.is_loading():
                    await asyncio.sleep(0.1)

            current_pos = await self.current_leader_client.body.position()

            leader_client_objective_xyz = await self.current_leader_client.quest_position.position()
            if await is_visible_by_path(self.current_leader_client, npc_range_path) and calc_Distance(leader_client_objective_xyz, current_pos) < 750.0:
                # await self.handle_interactibles(current_leader_client)

                # Handles interactables
                sigil_msg_check = await self.read_popup(self.current_leader_client)
                if "to enter" in sigil_msg_check.lower():
                    while 'to enter' in sigil_msg_check.lower():
                        logger.debug('Entering dungeon')
                        await asyncio.gather(*[p.send_key(Keycode.X, 0.1) for p in self.clients])
                        await asyncio.sleep(1.0)
                        for c in self.clients:
                            if await is_visible_by_path(c, dungeon_warning_path):
                                await c.send_key(Keycode.ENTER, 0.1)

                        sigil_msg_check = await self.read_popup(self.current_leader_client)

                    await self.handle_dungeon_entry(questing_friend_tp, follower_clients)
                else:
                    msg = sigil_msg_check.lower()
                    if 'to talk' in msg:
                        await asyncio.gather(*[p.send_key(Keycode.X, 0.1) for p in self.clients])
                        logger.debug('Talking to NPC')
                        await self.handle_npc_talking_quests(self.current_leader_client, self.clients)

                    elif 'magic raft' in msg or 'to ride' in msg or 'to teleport' in msg:
                        await self.current_leader_client.send_key(Keycode.X, 0.1)
                        await asyncio.sleep(1.0)
                        await asyncio.gather(*[p.send_key(Keycode.X, 0.1) for p in self.clients])
                    else:
                        await asyncio.gather(*[p.send_key(Keycode.X, 0.1) for p in self.clients])

                    # original_zone = await self.current_leader_client.zone_name()

                    await asyncio.sleep(2)
                    was_loading = False
                    for c in self.clients:
                        while await c.is_loading():
                            was_loading = True
                            await asyncio.sleep(0.1)

                        # try to correct for zone lag on follower clients by giving follower clients a second to get into the zone before teleporting
                        if was_loading:
                            await asyncio.sleep(1)

                    # Exit NPC menus (spell menus, quest menus, etc)
                    # await asyncio.sleep(2)
                    end_of_loop_paths = (exit_recipe_shop_path, exit_equipment_shop_path, cancel_multiple_quest_menu_path, cancel_spell_vendor, exit_snack_shop_path, exit_reagent_shop_path, exit_tc_vendor, exit_minigame_sigil, exit_wysteria_tournament, exit_dungeon_path, exit_zafaria_class_picture_button, exit_pet_leveled_up_button_path, avalon_badge_exit_button_path, potion_exit_path)
                    await asyncio.gather(*[exit_menus(c, end_of_loop_paths) for c in self.clients])

                    await asyncio.sleep(0.75)

                    if await is_visible_by_path(self.current_leader_client, spiral_door_teleport_path):
                        await self.handle_spiral_navigation()
            else:
                # we may be on a photomancy quest
                quest_objective = await get_quest_name(self.current_leader_client)

                if "Photomance" in quest_objective:
                    # Photomancy quests (WC, KM, LM)
                    await asyncio.gather(*[p.send_key(key=Keycode.Z, seconds=0.1) for p in self.clients])
                    await asyncio.gather(*[p.send_key(key=Keycode.Z, seconds=0.1) for p in self.clients])

            for c in self.clients:
                if await is_visible_by_path(c, missing_area_path):
                    # Handles when an area hasn't been downloaded yet
                    while not await is_visible_by_path(c, missing_area_retry_path):
                        await asyncio.sleep(0.1)
                    await click_window_by_path(c, missing_area_retry_path, True)

        await asyncio.sleep(0.7)

    async def handle_repeated_normal_quest_failures(self, last_leader_pid, last_leader_zone: str, iterations_since_last_quest_change: int):
        # In its current form, this attempts to correct for situations where you are standing too close to an NPC or sigil, and need to move away then return to get the popup to talk / enter
        # if after a certain number of loops we've failed to move on from our quest, something is wrong, and we try to correct for this in case this is the cause

        # this could be expanded to matching entity name to mob name for situations like the Labyrinth where the game tells you to defeat an enemy but gives you an inaccurate location

        # logger.info('ITERATIONS: ' + str(iterations_since_last_quest_change))
        if last_leader_pid == self.current_leader_client.process_id and last_leader_zone == await self.current_leader_client.zone_name():
            # more serious than 5 - the issue may be that the XYZ is just incorrect
            # we should scan for entities, teleport to the one that is closest to the given XYZ
            # if that fails, try the next one
            # if iterations_since_last_quest_change == ?:

            # first check - most likely scenario (and easiest to solve) is that we just need to move away and back towards the NPC or sigil
            if iterations_since_last_quest_change >= 5:
                location = await self.current_leader_client.body.position()
                await asyncio.gather(*[p.teleport(XYZ(location.x + 500, location.y, location.z - 1500)) for p in self.clients])
                await asyncio.sleep(2.0)

    async def hardcoded_collect(self, incompatible_hardcoded_quests, truncated_quest_obj: str):
        entity_data = incompatible_hardcoded_quests.get(truncated_quest_obj[:-1])
        for c in self.clients:
            if c.process_id != self.current_leader_client.process_id:
                current_quest = (await self.get_truncated_quest_objectives(c)).lower()
                follower_on_collect = False
                safe_location = await c.body.position()
                while current_quest == truncated_quest_obj:
                    follower_on_collect = True
                    for coord in entity_data[1:]:
                        try:
                            await c.teleport(coord)
                            await asyncio.sleep(1.0)

                            for i in range(3):
                                await c.send_key(Keycode.X)
                                await asyncio.sleep(.15)
                        except ValueError:
                            pass

                    current_quest = (await self.get_truncated_quest_objectives(c)).lower()

                if follower_on_collect:
                    while not await is_free(c) or c.entity_detect_combat_status:
                        await asyncio.sleep(1.0)

                    if await is_free(c) and not c.entity_detect_combat_status:
                        await c.teleport(safe_location)
                    logger.debug('Waiting for collect items to respawn.')
                    await asyncio.sleep((entity_data[0] + 5))

        current_quest = truncated_quest_obj
        safe_location = await self.current_leader_client.body.position()
        while current_quest == truncated_quest_obj:
            for coord in entity_data[1:]:
                try:
                    await self.current_leader_client.teleport(coord)
                    await asyncio.sleep(1.0)

                    for i in range(3):
                        await self.current_leader_client.send_key(Keycode.X)
                        await asyncio.sleep(.15)
                except ValueError:
                    pass

            current_quest = (await self.get_truncated_quest_objectives(self.current_leader_client)).lower()

        while not await is_free(self.current_leader_client) or self.current_leader_client.entity_detect_combat_status:
            await asyncio.sleep(1.0)

        if await is_free(self.current_leader_client) and not self.current_leader_client.entity_detect_combat_status:
            await self.current_leader_client.teleport(safe_location)

        entity_data = incompatible_hardcoded_quests.get(truncated_quest_obj[:-1])
        current_quest = truncated_quest_obj
        while current_quest == truncated_quest_obj:
            await asyncio.sleep(1.0)
            current_quest = (await self.get_truncated_quest_objectives(self.current_leader_client)).lower()

    async def handle_collect_quests(self):
        leader_obj = await self.get_truncated_quest_objectives(self.current_leader_client)
        # collect one item on all clients except leader
        for c in self.clients:
            if c.process_id != self.current_leader_client.process_id:
                follower_obj = await self.get_truncated_quest_objectives(c)

                if leader_obj == follower_obj:
                    collect_quester = Quester(c, self.clients, None)
                    await collect_quester.auto_collect_rewrite(c)

        # check if all clients have finished the entire quest
        # note this isnt guaranteed as auto_collect_rewrite only collects one item per call - it does not run until quest completion
        all_clients_moved_on = True
        for c in self.clients:
            if c.process_id != self.current_leader_client.process_id:
                follower_obj = await self.get_truncated_quest_objectives(c)
                if follower_obj == leader_obj:
                    all_clients_moved_on = False

        # finally, collect items on the leader
        # only do this if all clients have already finished their own collects
        if all_clients_moved_on:
            await self.auto_collect_rewrite(self.current_leader_client)

    async def bring_clients_to_same_location(self, questing_friend_tp: bool, gear_switching_in_solo_zones: bool):
        leader_pos = await self.current_leader_client.body.position()
        teleported = False
        for c in self.clients:
            if await c.zone_name() == await self.current_leader_client.zone_name():
                errored = True
                # teleport throws should update bool
                while errored:
                    try:
                        await c.teleport(leader_pos)
                        errored = False
                        teleported = True
                    except ValueError:
                        errored = True
                        await asyncio.sleep(1.0)

        if teleported:
            await asyncio.sleep(2.0)

        # only friend TPs if clients are in different zones or we think we may be in a solo zone
        if questing_friend_tp:
            maybe_solo_zone = await self.determine_solo_zone()
            await self.zone_recorrect_friend_tp(maybe_solo_zone=maybe_solo_zone, gear_switching_in_solo_zones=gear_switching_in_solo_zones)

    async def initial_auto_pet(self, energy_info, questing_clients: list[Client], ignore_pet_level_up: bool, play_dance_game: bool):
        auto_pet_on = False
        for p in self.clients:
            if p.auto_pet_status:
                auto_pet_on = True

        if auto_pet_on:
            # potentially run auto_pet once initially, if all questing_clients have high energy
            all_high_energy = True
            for i, c in enumerate(questing_clients):
                # current energy / total energy - percent of remaining energy
                if ((energy_info[i][0] / energy_info[i][1]) * 100) < 70:
                    all_high_energy = False

            if all_high_energy:
                # buy potions if necessary, otherwise auto pet will fail
                await self.heal_and_handle_potions()

                logger.debug('All questing clients have high energy, training pets on all clients.')
                await asyncio.gather(*[auto_pet(c, ignore_pet_level_up, play_dance_game, questing=True) for c in self.clients])

    async def leader_wait_for_free(self, p: Client):
        while not await is_free_leader_questing(p):
            await asyncio.sleep(.1)

    # TODO: Slay the beast
    async def auto_quest_leader(self, questing_friend_tp: bool, gear_switching_in_solo_zones: bool, hitting_client, ignore_pet_level_up: bool, play_dance_game: bool):
        follower_clients = await self.get_follower_clients()
        questing_clients = await self.get_questing_clients()

        # read and store the name of the client's wizard, and check energy
        if questing_friend_tp or self.current_leader_client.auto_pet_status:
            # open character screen
            await asyncio.gather(*[self.open_character_screen(c) for c in self.clients])

            await asyncio.gather(*[set_wizard_name_from_character_screen(c) for c in self.clients])
            energy_info = await asyncio.gather(*[return_wizard_energy_from_character_screen(c) for c in questing_clients])

            # close character screen
            await asyncio.gather(*[self.close_character_screen(c) for c in self.clients])

        # if all questing clients have high energy, run auto pet once initially
        if self.current_leader_client.auto_pet_status:
            await self.initial_auto_pet(energy_info, questing_clients, ignore_pet_level_up, play_dance_game)

        # gather all clients to the same zone and exact location
        await self.bring_clients_to_same_location(questing_friend_tp, gear_switching_in_solo_zones)

        # check for solo zone again as we may have changed zones
        maybe_solo_zone = await self.determine_solo_zone()

        # leader and follower clients can dynamically change during auto questing to account for clients being left behind
        client_quests = await self.get_client_quests(questing_clients=questing_clients)

        # just for providing info to the user in the log statement
        if len(client_quests) > 0:
            s = ''
            for cl in client_quests:
                if s == '':
                    s = cl.title
                else:
                    s = s + ', ' + cl.title

            logger.info('Clients on same quest: ' + s)

        # information needed for handle_repeated_normal_quest_failures()
        iterations_since_last_quest_change = 0
        leader_last_full_quest = await get_quest_name(self.current_leader_client)
        last_leader_pid = self.current_leader_client.process_id
        last_leader_zone = await self.current_leader_client.zone_name()

        # main loop
        while self.client.questing_status:
            await asyncio.sleep(.4)

            # in case client(s) in combat and the questing loop continued anyway
            await asyncio.gather(*[self.leader_wait_for_free(p) for p in self.clients])

            # Collect wisps, use potions, or get potions if necessary
            await self.heal_and_handle_potions()

            # if dungeon recall button is visible, click it
            await self.handle_dungeon_recall(follower_clients=follower_clients)

            # if auto pet is enabled and all questing clients have leveled up, send all to pavilion and train pets on ALL clients
            await self.auto_pet_questing(questing_clients, ignore_pet_level_up, play_dance_game)

            # dynamically change leader client when follower's get left behind
            # if there were previously clients on the same quest check for quest objective change on all clients
            follower_clients, client_quests = await self.determine_new_leader_and_followers(client_quests, questing_clients, follower_clients)

            # handle circumstances where any follower client is not in the same zone as the leader client
            # keep in mind, the previous leader client may now be a follower client since we have just called determine_new_leader_and_followers()
            await self.handle_zone_correction(maybe_solo_zone, questing_friend_tp, gear_switching_in_solo_zones)

            if await is_free_leader_questing(self.current_leader_client):
                quest_xyz = await self.current_leader_client.quest_position.position()
                distance = calc_Distance(quest_xyz, XYZ(0.0, 0.0, 0.0))

                # we are almost certainly not on a collect quest
                if distance > 1:
                    # attempt to fix cases where we loop through several times without completing a quest
                    await self.handle_repeated_normal_quest_failures(last_leader_pid, last_leader_zone, iterations_since_last_quest_change)

                    await self.teleport_to_quest(hitting_client, follower_clients)

                    await self.handle_normal_quests(follower_clients, questing_friend_tp)
                else:
                    # Double check - sometimes wiz lies about quest position - a simple sleep and re-grabbing of the quest xyz solves the issue
                    await asyncio.sleep(3.0)
                    quest_xyz = await self.current_leader_client.quest_position.position()
                    distance = calc_Distance(quest_xyz, XYZ(0.0, 0.0, 0.0))

                    if distance < 1:
                        quest_objective = await get_quest_name(self.current_leader_client)

                        truncated_quest_obj = (await self.get_truncated_quest_objectives(self.current_leader_client)).lower()
                        # Key - truncated quest name
                        # Values - 0: time between entity respawns, 1-... : xyz locations of separate entities belonging to the quest
                        # TODO: Get rid of this once scripting system allows for quest libraries
                        incompatible_hardcoded_quests = {
                            'find submarine parts in the floating land': [40, XYZ(x=-17412.53515625, y=10505.794921875, z=-429.19927978515625), XYZ(x=-17088.447265625, y=12284.244140625, z=-410.1100769042969), XYZ(x=-21681.78125, y=11547.5966796875, z=-429.19927978515625)]
                            , 'collect sea cucumbers in pitch black lake': [40, XYZ(x=-8995.333984375, y=-830.6781616210938, z=-97.31965637207031), XYZ(x=-9911.9404296875, y=823.7628173828125, z=-97.31971740722656), XYZ(x=-11497.998046875, y=-1030.2618408203125, z=-97.32334899902344), XYZ(x=-12099.0517578125, y=-4674.13623046875, z=-96.95536804199219), XYZ(x=-14215.37109375, y=-4371.93798828125, z=-97.32661437988281), XYZ(x=-14287.5556640625, y=-2835.4814453125, z=-97.3197021484375)]
                            , 'catch vonda fish in pitch black lake': [3, XYZ(x=-10030.9912109375, y=-7925.51953125, z=-347.3206787109375)]
                            , 'break mining equipment in tyrian gorge': [25, XYZ(x=-2529.12890625, y=-3015.832763671875, z=-906.33837890625), XYZ(x=-3181.15234375, y=-6233.39306640625, z=-924.6146240234375), XYZ(x=1495.8695068359375, y=-7369.19580078125, z=-905.3265380859375), XYZ(x=-204.098876953125, y=-8468.421875, z=236.10702514648438), XYZ(x=2508.4169921875, y=-4712.10302734375, z=-1232.338134765625)]
                            , 'steal barrel of kermes fire in tyrian gorge': [35, XYZ(x=1352.575927734375, y=-4787.16552734375, z=-1098.1168212890625), XYZ(x=-1936.0185546875, y=-3954.15380859375, z=-1077.83837890625), XYZ(x=-1820.606201171875, y=-6331.4541015625, z=-1077.849853515625), XYZ(x=517.3740844726562, y=-7042.94287109375, z=-1077.84130859375), XYZ(x=-4628.16650390625, y=-5603.2578125, z=-906.51318359375), XYZ(x=-3216.014892578125, y=-5965.7412109375, z=-932.0715942382812), XYZ(x=-1106.3753662109375, y=-3800.721923828125, z=-905.3256225585938), XYZ(x=-451.9595031738281, y=-4547.75537109375, z=-905.325927734375)]
                        }

                        if any(map(truncated_quest_obj.__contains__, incompatible_hardcoded_quests.keys())):
                            logger.debug('Hardcoded collect')
                            await self.hardcoded_collect(incompatible_hardcoded_quests, truncated_quest_obj)
                        # normal collect quest
                        else:
                            logger.debug('Completing collect quest')
                            await self.handle_collect_quests()

                    else:
                        logger.debug('False collect quest detected.  Quest position: ' + str(distance))

            # keep track of how many loops we've gone through without changing our quest
            # if leader happened to change, reset the counter
            leader_full_current_quest = await get_quest_name(self.current_leader_client)
            # quest hasn't changed, new leader has not been assigned, and zone hasn't changed
            # this means we have failed to complete the last single quest objective
            if leader_full_current_quest == leader_last_full_quest and last_leader_pid == self.current_leader_client.process_id and last_leader_zone == await self.current_leader_client.zone_name():
                iterations_since_last_quest_change += 1
            else:
                leader_last_full_quest = await get_quest_name(self.current_leader_client)
                last_leader_pid = self.current_leader_client.process_id
                last_leader_zone = await self.current_leader_client.zone_name()
                iterations_since_last_quest_change = 0

    async def handle_questing_zone_change(self):
        if await is_visible_by_path(self.client, exit_dungeon_path):
            async with self.client.mouse_handler:
                await asyncio.sleep(1.0)
                await click_window_by_path(self.client, exit_dungeon_path)
                await self.client.wait_for_zone_change()
                await asyncio.sleep(1.0)
        else:
            while await self.client.is_loading():
                await asyncio.sleep(.1)

        # Track entities after zone change
        from src.quest_db import get_quest_db
        db = get_quest_db()
        await db.track_zone_entities(self.client)

    async def auto_quest_solo(self, auto_pet_disabled=False, ignore_pet_level_up=False, play_dance_game=False):
        if await is_free(self.client):
            if await is_potion_needed(self.client) and await self.client.stats.current_mana() > 1 and await self.client.stats.current_hitpoints() > 1:
                await collect_wisps(self.client)

            if self.client.use_potions:
                await auto_potions(self.client, True, buy=self.client.buy_potions)

            quest_xyz = await self.client.quest_position.position()

            if self.client.auto_pet_status and not auto_pet_disabled:
                # client has leveled up
                if self.client.character_level < await self.client.stats.reference_level():
                    logger.debug('Client ' + self.client.title + ' leveled up - training pet.')
                    await auto_pet(self.client, ignore_pet_level_up, play_dance_game, questing=True)

            distance = calc_Distance(quest_xyz, XYZ(0.0, 0.0, 0.0))
            if distance > 1:
                while self.client.entity_detect_combat_status:
                    await asyncio.sleep(.1)

                await navmap_tp(self.client, quest_xyz)

                # confirm exit dungeon early button or wait for client to exit loading
                await self.handle_questing_zone_change()

                await asyncio.sleep(.5)
                if await is_visible_by_path(self.client, cancel_chest_roll_path):
                    # Handles chest reroll menu, will always cancel
                    await click_window_by_path(self.client, cancel_chest_roll_path)

                current_pos = await self.client.body.position()
                if await is_visible_by_path(self.client, npc_range_path) and calc_Distance(quest_xyz, current_pos) < 750.0:
                    # Handles interactables
                    sigil_msg_check = await self.read_popup(self.client)
                    if "to enter" in sigil_msg_check.lower():
                        # Handles entering dungeons
                        await asyncio.gather(*[p.send_key(Keycode.X, 0.1) for p in self.clients])
                        for c in self.clients:
                            while not await c.is_loading():
                                if await is_visible_by_path(c, dungeon_warning_path):
                                    await c.send_key(Keycode.ENTER, 0.1)
                                await asyncio.sleep(0.1)

                        for c in self.clients:
                            while await c.is_loading():
                                await asyncio.sleep(0.1)
                    elif 'to talk' in sigil_msg_check.lower():
                        logger.debug('Talking to NPC')
                        await self.client.send_key(Keycode.X, 0.1)
                        await self.handle_npc_talking_quests(self.client, [self.client])

                    else:
                        await self.client.send_key(Keycode.X, 0.1)

                        await asyncio.sleep(0.75)
                        if await is_visible_by_path(self.client, spiral_door_teleport_path):
                            # Handles spiral door navigation
                            if await self.new_world_doors(self.client) == False:
                                await spiral_door_with_quest(self.client)

                quest_objective = await get_quest_name(self.client)

                if "Photomance" in quest_objective:
                    # Photomancy quests (WC, KM, LM)
                    await self.client.send_key(key=Keycode.Z, seconds=0.1)
                    await self.client.send_key(key=Keycode.Z, seconds=0.1)

                if await is_visible_by_path(self.client, missing_area_path):
                    # Handles when an area hasn't been downloaded yet
                    while not await is_visible_by_path(self.client, missing_area_retry_path):
                        await asyncio.sleep(0.1)
                    await click_window_by_path(self.client, missing_area_retry_path, True)

            else:
                # Double check - sometimes wiz lies about quest position - a simple sleep and re-grabbing of the quest xyz seems to solve the issue
                await asyncio.sleep(3.0)
                quest_xyz = await self.client.quest_position.position()
                distance = calc_Distance(quest_xyz, XYZ(0.0, 0.0, 0.0))

                if distance < 1:
                    await self.auto_collect_rewrite(self.client)

    async def auto_quest(self, ignore_pet_level_up: bool, play_dance_game: bool):
        while self.client.questing_status:
            await asyncio.sleep(1)
            await self.auto_quest_solo(ignore_pet_level_up=ignore_pet_level_up, play_dance_game=play_dance_game)



async def is_free_leader_questing(client: Client):
    # Returns True if not in combat, loading screen, dialogue, or forced animation dialogue.
    dialogue_text = await read_dialogue_text(client)
    return not any([await client.is_loading(), await client.in_battle(), await is_visible_by_path(client, advance_dialog_path), client.entity_detect_combat_status, (dialogue_text != '')])


async def read_dialogue_text(p: Client) -> str:
    try:
        dialogue_text = await get_window_from_path(p.root_window, dialog_text_path)
        txtmsg = await dialogue_text.maybe_text()
    except:
        txtmsg = ''
    return txtmsg