from enum import Enum, auto
import gettext
import queue
import re
import PySimpleGUI as gui
import pyperclip
from src.combat_objects import school_id_to_names
from src.paths import wizard_city_dance_game_path
from src.utils import assign_pet_level, get_ui_tree_text
from threading import Thread

import sys
import re
from loguru import logger
import ctypes

gui.set_global_icon("..\\Deimos-logo.ico")
gui.set_options(suppress_error_popups=True, suppress_raise_key_errors=True)
gui.PySimpleGUI.SUPPRESS_ERROR_POPUPS = True
gui.PySimpleGUI.SUPPRESS_RAISE_KEY_ERRORS = True

global console_sink

# from Deimos import show_expanded_logs


import ctypes

def terminate_thread(thread: Thread):
    if not thread.is_alive():
        return
    
    exc = ctypes.py_object(SystemExit)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_long(thread.ident), exc)
    if res == 0:
        raise ValueError("Invalid thread ID")
    elif res != 1:
        # If more than one thread was affected, something went wrong
        ctypes.pythonapi.PyThreadState_SetAsyncExc(thread.ident, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")

class ToolClosedException(Exception):
    pass

class PsgSink:
    def __init__(self, window: gui.Window, key):
        self.window: gui.Window = window
        self.key = key
        self.buffer = []
        self.max_lines = 1000  # Prevent excessive memory usage
        self.show_expanded_logs = False

        # Define color mapping for different log levels
        self.level_colors = {
            "DEBUG": "grey",
            "INFO": "white",
            "SUCCESS": "white",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "white"
        }

        self.level_bg = {
            "SUCCESS": "green",
            "CRITICAL": "red"
        }

    def copy(self):
        log_str = "```\n"
        # print(1)
        for (line, _, _) in self.buffer:
            log_str += line

        pyperclip.copy(log_str + "```")
        logger.debug("Copied current logs.")

    def toggle_show_expanded_logs(self, override: bool | None = None):
        match override:
            case True | False:
                self.show_expanded_logs = override
            case _:
                self.show_expanded_logs = not self.show_expanded_logs

        match self.show_expanded_logs:
            case True:
                logger.debug("Console is now showing full log messages.")
            case _:
                logger.debug("Console is now truncating log messages.")

        self.refresh()

    def write(self, message):
        # Determine log level from message
        # text_color = "white"  # Default color

        # Strip ANSI color codes
        ansi_pattern = r'\033\[\d+m'
        clean_message = re.sub(ansi_pattern, '', message)

        split_msg = clean_message.split("|")
        if len(split_msg) < 3:
            for l, c in self.level_colors.items():
                if l in clean_message:
                    level = l
                    break
            else:
                level = "DEBUG"
        
        else:
            level = split_msg[1].lstrip().rstrip()

        def collapse_log(input: str) -> str:
            if "-" not in input:
                return input
            
            split_input = input.split("-")
            if len(split_input) < 4:
                return input
            
            return split_input[3].lstrip()

        truncated_message = level + " - " + collapse_log(clean_message)

        # Add to buffer and trim if needed
        self.buffer.append((clean_message, truncated_message, level))
        if len(self.buffer) > self.max_lines:
            self.buffer.pop(0)

        try:
            # Add the message with appropriate color
            message_to_write = clean_message
            if not self.show_expanded_logs:
                message_to_write = truncated_message

            if level in self.level_bg:
                self.window[self.key].print(message_to_write, end='', text_color=self.level_colors[level], background_color = self.level_bg[level])
                return
            
            self.window[self.key].print(message_to_write, end='', text_color=self.level_colors[level])
        except Exception:
            # Silently fail if we can't print to the window
            pass

    def refresh(self):
        self.window[self.key].update('')
        for clean, trunc, level in self.buffer:
            try:
                message_to_write = clean
                if not self.show_expanded_logs:
                    message_to_write = trunc

                if level in self.level_bg:
                    self.window[self.key].print(message_to_write, end='', text_color=self.level_colors[level], background_color = self.level_bg[level])
                    continue
                
                self.window[self.key].print(message_to_write, end='', text_color=self.level_colors[level])
            except Exception:
                pass

    def get_buffer(self):
        return self.buffer


class GUICommandType(Enum):
    # deimos <-> window
    Close = auto()
    AttemptedClose = auto()
    CloseFromBackend = auto()

    # window -> deimos
    ToggleOption = auto()
    Copy = auto()
    SelectEnemy = auto()

    Teleport = auto()
    CustomTeleport = auto()
    EntityTeleport = auto()

    XYZSync = auto()
    XPress = auto()

    GoToZone = auto()
    GoToWorld = auto()
    GoToBazaar = auto()

    RefillPotions = auto()

    AnchorCam = auto()
    SetCamPosition = auto()
    SetCamDistance = auto()

    ExecuteFlythrough = auto()
    KillFlythrough = auto()

    ExecuteBot = auto()
    KillBot = auto()

    SetPlaystyles = auto()

    # SetPetWorld = auto()

    SetScale = auto()

    # deimos -> window
    UpdateWindow = auto()
    UpdateWindowValues = auto()
    UpdateConsole = auto()
    CopyConsole = auto()

    GoToEntity = auto()
    RefreshEntities = auto()

    ShowUITreePopup = auto()
    ShowEntityListPopup = auto()

# TODO:
# - inherit from StrEnum in 3.11 to make this nicer
# - fix naming convention, it's inconsistent
class GUIKeys:
    toggle_speedhack = "togglespeedhack"
    toggle_combat = "togglecombat"
    toggle_dialogue = "toggledialogue"
    toggle_sigil = "togglesigil"
    toggle_questing = "toggle_questing"
    toggle_auto_pet = "toggleautopet"
    toggle_auto_potion = "toggleautopotion"
    toggle_freecam = "togglefreecam"
    toggle_camera_collision = "togglecameracollision"
    toggle_show_expanded_logs = "toggleshowexpandedlogs"

    hotkey_quest_tp = "hotkeyquesttp"
    hotkey_freecam_tp = "hotkeyfreecamtp"

    mass_hotkey_mass_tp = "masshotkeymasstp"
    mass_hotkey_xyz_sync = "masshotkeyxyzsync"
    mass_hotkey_x_press = "masshotkeyxpress"

    copy_position = "copyposition"
    copy_zone = "copyzone"
    copy_rotation = "copyrotation"
    copy_entity_list = "copyentitylist"
    copy_ui_tree = "copyuitree"
    copy_camera_position = "copycameraposition"
    copy_stats = "copystats"
    copy_camera_rotation = "copycamerarotation"
    copy_logs = "copylogs"

    button_custom_tp = "buttoncustomtp"
    button_entity_tp = "buttonentitytp"
    button_go_to_zone = "buttongotozone"
    button_mass_go_to_zone = "buttonmassgotozone"
    button_go_to_world = "buttongotoworld"
    button_mass_go_to_world = "buttonmassgotoworld"
    button_go_to_bazaar = "buttongotobazaar"
    button_mass_go_to_bazaar = "buttonmassgotobazaar"
    button_refill_potions = "buttonrefillpotions"
    button_mass_refill_potions = "buttonmassrefillpotions"
    button_set_camera_position = "buttonsetcameraposition"
    button_anchor = "buttonanchor"
    button_set_distance = "buttonsetdistance"
    button_view_stats = "buttonviewstats"
    button_swap_members = "buttonswapmembers"

    button_go_to_entity = "buttongotoentity"
    button_mass_go_to_entity = "buttonmassgotoentity"
    button_refresh_entities = "buttonrefreshentities"

    button_execute_flythrough = "buttonexecuteflythrough"
    button_kill_flythrough = "buttonkillflythrough"
    button_run_bot = "buttonrunbot"
    button_kill_bot = "buttonkillbot"
    button_set_playstyles = "buttonsetplaystyles"
    button_set_scale = "buttonsetscale"


class GUICommand:
    def __init__(self, com_type: GUICommandType, data=None):
        self.com_type = com_type
        self.data = data


def hotkey_button(name: str, key, auto_size: bool, text_color: str, button_color: str):
    return gui.Button(name, button_color=(text_color, button_color), auto_size_button=auto_size, key=key)


def create_gui(gui_theme, gui_text_color, gui_button_color, tool_name, tool_version, gui_on_top, langcode):
    gui.theme(gui_theme)

    if langcode != 'en':
        translate = gettext.translation("messages", "locale", languages=[langcode])
        tl = translate.gettext
    else:
        # maybe use gettext (module) as translate instead?
        gettext.bindtextdomain('messages', 'locale')
        gettext.textdomain('messages')
        tl = gettext.gettext

    gui.popup(
        tl('Deimos will always be free and open-source.\nBy using Deimos, you agree to the GPL v3 license agreement.\nIf you bought this, you got scammed!'), 
        title=tl('License Agreement'), 
        keep_on_top=True, 
        text_color=gui_text_color, 
        button_color=(gui_text_color, gui_button_color),
        auto_close = True,
        auto_close_duration = 5,
        non_blocking = True
    )

    global hotkey_button
    original_hotkey_button = hotkey_button

    def hotkey_button(name, key, auto_size=False, text_color=gui_text_color, button_color=gui_button_color):
        return original_hotkey_button(name, key, auto_size, text_color, button_color)

    # TODO: Switch to using keys for this stuff
    toggles: list[tuple[str, str]] = [
        (tl('Speedhack'), GUIKeys.toggle_speedhack),
        (tl('Combat'), GUIKeys.toggle_combat),
        (tl('Dialogue'), GUIKeys.toggle_dialogue),
        (tl('Sigil'), GUIKeys.toggle_sigil),
        (tl('Questing'), GUIKeys.toggle_questing),
        (tl('Auto Pet'), GUIKeys.toggle_auto_pet),
        (tl('Auto Potion'), GUIKeys.toggle_auto_potion),
    ]
    hotkeys: list[tuple[str, str]] = [
        (tl('Quest TP'), GUIKeys.hotkey_quest_tp),
        (tl('Freecam'), GUIKeys.toggle_freecam),
        (tl('Freecam TP'), GUIKeys.hotkey_freecam_tp)
    ]
    mass_hotkeys = [
        (tl('Mass TP'), GUIKeys.mass_hotkey_mass_tp),
        (tl('XYZ Sync'), GUIKeys.mass_hotkey_xyz_sync),
        (tl('X Press'), GUIKeys.mass_hotkey_x_press)
    ]
    toggles_layout = [[hotkey_button(name, key), gui.Text(tl('Disabled'), key=f'{name}Status', auto_size_text=False, size=(7, 1), text_color=gui_text_color)] for name, key in toggles]
    framed_toggles_layout = gui.Frame(tl('Toggles'), toggles_layout, title_color=gui_text_color)
    hotkeys_layout = [[hotkey_button(name, key)] for name, key in hotkeys]
    framed_hotkeys_layout = gui.Frame(tl('Hotkeys'), hotkeys_layout, title_color=gui_text_color)
    mass_hotkeys_layout = [[hotkey_button(name, key)] for name, key in mass_hotkeys]
    framed_mass_hotkeys_layout = gui.Frame(tl('Mass Hotkeys'), mass_hotkeys_layout, title_color=gui_text_color)

    client_title = gui.Text(tl('Client') + ': ', key='Title', text_color=gui_text_color)

    # TODO: Does it make any sense to translate this? Has more occurences later in the file
    # x_pos = gui.Text('x: ', key='x', auto_size_text=False, text_color=gui_text_color)
    # y_pos = gui.Text('y: ', key='y', auto_size_text=False, text_color=gui_text_color)
    # z_pos = gui.Text('z: ', key='z', auto_size_text=False, text_color=gui_text_color)
    # yaw = gui.Text(tl('Yaw') + ': ', key='Yaw', auto_size_text=False, text_color=gui_text_color)
    xyz_pos = gui.Text("Position (XYZ): ", key = 'xyz', auto_size_text=False, text_color=gui_text_color)
    ypr_ori = gui.Text("Orientation (PRY): ", key = 'pry', auto_size_text=False, text_color=gui_text_color)

    zone_info = gui.Text(tl('Zone') + ': ', key='Zone', auto_size_text=False, size=(62, 1), text_color=gui_text_color)

    copy_pos = hotkey_button(tl('Copy Position'), GUIKeys.copy_position)
    copy_zone = hotkey_button(tl('Copy Zone'), GUIKeys.copy_zone)
    copy_yaw = hotkey_button(tl('Copy Rotation'), GUIKeys.copy_rotation)

    client_info_layout = [
        [client_title],
        [zone_info],
        # [x_pos],
        # [y_pos],
        # [z_pos],
        # [yaw]
        [xyz_pos],
        [ypr_ori]
    ]

    utils_layout = [
        [copy_zone],
        [copy_pos],
        [copy_yaw]
    ]

    framed_utils_layout = gui.Frame(tl('Utils'), utils_layout, title_color=gui_text_color)

    dev_utils_notice = tl('The utils below are for advanced users and no support will be given on them.')
    helpful_support_notice = tl('Be sure to include your logs when asking for support.')

    custom_tp_layout = [
        [gui.Text(dev_utils_notice, text_color=gui_text_color)],
        [
            gui.Text('X:', text_color=gui_text_color), gui.InputText(size=(6, 1), key='XInput'),
            gui.Text('Y:', text_color=gui_text_color), gui.InputText(size=(6, 1), key='YInput'),
            gui.Text('Z:', text_color=gui_text_color), gui.InputText(size=(7, 1), key='ZInput'),
            gui.Text(tl('Yaw') + ': ', text_color=gui_text_color), gui.InputText(size=(6, 1), key='YawInput'),
            hotkey_button(tl('Custom TP'), GUIKeys.button_custom_tp)
        ],
        [
            gui.Text(tl('Entity Name') + ':', text_color=gui_text_color), gui.InputText(size=(36, 1), key='EntityTPInput'),
            hotkey_button(tl('Entity TP'), GUIKeys.button_entity_tp)
        ]
    ]

    framed_custom_tp_layout = gui.Frame(tl('TP Utils'), custom_tp_layout, title_color=gui_text_color)

    dev_utils_layout = [
        [gui.Text(dev_utils_notice, text_color=gui_text_color)],
        [
            hotkey_button(tl('Available Entities'), GUIKeys.copy_entity_list, True),
            hotkey_button(tl('Available Paths'), GUIKeys.copy_ui_tree, True)
        ],
        [
            gui.Text(tl('Zone Name') + ':', text_color=gui_text_color), gui.InputText(size=(13, 1), key='ZoneInput'),
            hotkey_button(tl('Go To Zone'), GUIKeys.button_go_to_zone),
            hotkey_button(tl('Mass Go To Zone'), GUIKeys.button_mass_go_to_zone, True)
        ],
        [
            gui.Text(tl('World Name') + ':', text_color=gui_text_color),
            # TODO: Come back with some ingenius solution for this
            gui.Combo(
                ['WizardCity', 'Krokotopia', 'Marleybone', 'MooShu', 'DragonSpire', 'Grizzleheim', 'Celestia', 'Wysteria', 'Zafaria', 'Avalon', 'Azteca', 'Khrysalis', 'Polaris', 'Mirage', 'Empyrea', 'Karamelle', 'Lemuria'],
                default_value='WizardCity', readonly=True,text_color=gui_text_color, size=(13, 1), key='WorldInput'
            ),
            hotkey_button(tl('Go To World'), GUIKeys.button_go_to_world, True),
            hotkey_button(tl('Mass Go To World'), GUIKeys.button_mass_go_to_world, True)
        ],
        [
            hotkey_button(tl('Go To Bazaar'), GUIKeys.button_go_to_bazaar, True),
            hotkey_button(tl('Mass Go To Bazaar'), GUIKeys.button_mass_go_to_bazaar, True),
            hotkey_button(tl('Refill Potions'), GUIKeys.button_refill_potions, True),
            hotkey_button(tl('Mass Refill Potions'), GUIKeys.button_mass_refill_potions, True)
        ]
    ]

    framed_dev_utils_layout = gui.Frame(tl('Dev Utils'), dev_utils_layout, title_color=gui_text_color)

    camera_controls_layout = [
        [gui.Text(dev_utils_notice, text_color=gui_text_color)],
        [
            gui.Text('X:', text_color=gui_text_color), gui.InputText(size=(10, 1), key='CamXInput'),
            gui.Text('Y:', text_color=gui_text_color), gui.InputText(size=(10, 1), key='CamYInput'),
            gui.Text('Z:', text_color=gui_text_color), gui.InputText(size=(10, 1), key='CamZInput'),
            hotkey_button(tl('Set Camera Position'), GUIKeys.button_set_camera_position, True)
        ],
        [
            gui.Text(tl('Yaw') + ':', text_color=gui_text_color), gui.InputText(size=(10, 1), key='CamYawInput'),
            gui.Text(tl('Roll') + ':', text_color=gui_text_color), gui.InputText(size=(10, 1), key='CamRollInput'),
            gui.Text(tl('Pitch') + ':', text_color=gui_text_color), gui.InputText(size=(10, 1), key='CamPitchInput')
        ],
        [
            gui.Text(tl('Entity') + ':', text_color=gui_text_color), gui.InputText(size=(18, 1), key='CamEntityInput'),
            hotkey_button(tl('Anchor'), GUIKeys.button_anchor, text_color=gui_text_color),
            hotkey_button(tl('Toggle Camera Collision'), GUIKeys.toggle_camera_collision, True)
        ],
        [
            gui.Text(tl('Distance') + ':', text_color=gui_text_color), gui.InputText(size=(10, 1), key='CamDistanceInput'),
            gui.Text(tl('Min') + ':', text_color=gui_text_color), gui.InputText(size=(10, 1), key='CamMinInput'),
            gui.Text(tl('Max') + ':', text_color=gui_text_color), gui.InputText(size=(10, 1), key='CamMaxInput'),
            hotkey_button(tl('Set Distance'), GUIKeys.button_set_distance, True)
        ],
        [
            hotkey_button(tl('Copy Camera Position'), GUIKeys.copy_camera_position, True),
            hotkey_button(tl('Copy Camera Rotation'), GUIKeys.copy_camera_rotation, True)
        ]
    ]

    framed_camera_controls_layout = gui.Frame(tl('Camera Controls'), camera_controls_layout, title_color=gui_text_color)

    stat_viewer_layout = [
        [gui.Text(dev_utils_notice, text_color=gui_text_color)],
        [gui.Text(tl('Caster/Target Indices') + ':', text_color=gui_text_color), gui.Combo([i + 1 for i in range(12)], text_color=gui_text_color, size=(21, 1), default_value=1, key='EnemyInput', readonly=True), gui.Combo([i + 1 for i in range(12)], text_color=gui_text_color, size=(21, 1), default_value=1, key='AllyInput', readonly=True)],
        [
            gui.Text(tl('Dmg') + ':', text_color=gui_text_color), gui.InputText('', size=(7, 1), key='DamageInput'),
            gui.Text(tl('School') + ':', text_color=gui_text_color),
            # TODO: Also needs some smart solution
            gui.Combo(['Fire', 'Ice', 'Storm', 'Myth', 'Life', 'Death', 'Balance', 'Star', 'Sun', 'Moon', 'Shadow'], default_value='Fire', size=(7, 1), key='SchoolInput', readonly=True),
            gui.Text(tl('Crit') + ':', text_color=gui_text_color), gui.Checkbox(None, True, text_color=gui_text_color, key='CritStatus'),
            hotkey_button(tl('View Stats'), GUIKeys.button_view_stats, True),
            hotkey_button(tl('Copy Stats'), GUIKeys.copy_stats, True)
        ],
        [gui.Multiline(tl('No client has been selected.'), key='stat_viewer', size=(64, 8), text_color=gui_text_color, horizontal_scroll=True)],
        [
            hotkey_button(tl('Swap Members'), GUIKeys.button_swap_members, True),
            gui.Text(tl('Force School Damage') + ':', text_color=gui_text_color),
            gui.Checkbox(None, text_color=gui_text_color, key='ForceSchoolStatus')
        ],
    ]

    framed_stat_viewer_layout = gui.Frame(tl('Stat Viewer'), stat_viewer_layout, title_color=gui_text_color)

    flythrough_layout = [
        [gui.Text(dev_utils_notice, text_color=gui_text_color)],
        [gui.Multiline(key='flythrough_creator', size=(64, 11), text_color=gui_text_color, horizontal_scroll=True)],
        [
            gui.Input(key='flythrough_file_path', visible=False),
            gui.FileBrowse(tl('Import Flythrough'), file_types=(("Text Files", "*.txt"),), auto_size_button=True, button_color=(gui_text_color, gui_button_color)),
            gui.Input(key='flythrough_save_path', visible=False),
            gui.FileSaveAs(tl('Export Flythrough'), file_types=(("Text Files", "*.txt"),), auto_size_button=True, button_color=(gui_text_color, gui_button_color)),
            hotkey_button(tl('Execute Flythrough'), GUIKeys.button_execute_flythrough, True),
            hotkey_button(tl('Kill Flythrough'), GUIKeys.button_kill_flythrough, True)
            ],
    ]

    framed_flythrough_layout = gui.Frame(tl('Flythrough Creator'), flythrough_layout, title_color=gui_text_color)

    bot_creator_layout = [
        [gui.Text(dev_utils_notice, text_color=gui_text_color)],
        [gui.Multiline(key='bot_creator', size=(64, 11), text_color=gui_text_color, horizontal_scroll=True)],
        [
            gui.Input(key='bot_file_path', visible=False),
            gui.FileBrowse('Import Bot', file_types=(("Text Files", "*.txt"),), auto_size_button=True, button_color=(gui_text_color, gui_button_color)),
            gui.Input(key='bot_save_path', visible=False),
            gui.FileSaveAs('Export Bot', file_types=(("Text Files", "*.txt"),), auto_size_button=True, button_color=(gui_text_color, gui_button_color)),
            hotkey_button(tl('Run Bot'), GUIKeys.button_run_bot, True),
            hotkey_button(tl('Kill Bot'), GUIKeys.button_kill_bot, True)
            ],
    ]

    framed_bot_creator_layout = gui.Frame(tl('Bot Creator'), bot_creator_layout, title_color=gui_text_color)

    combat_config_layout = [
        [gui.Text(dev_utils_notice, text_color=gui_text_color)],
        [gui.Multiline(key='combat_config', size=(64, 11), text_color=gui_text_color, horizontal_scroll=True)],
        [
            gui.Input(key='combat_file_path', visible=False),
            gui.FileBrowse('Import Playstyle', file_types=(("Text Files", "*.txt"),), auto_size_button=True, button_color=(gui_text_color, gui_button_color)),
            gui.Input(key='combat_save_path', visible=False),
            gui.FileSaveAs('Export Playstyle', file_types=(("Text Files", "*.txt"),), auto_size_button=True, button_color=(gui_text_color, gui_button_color)),
            hotkey_button(tl('Set Playstyles'), GUIKeys.button_set_playstyles, True),
        ],
    ]

    framed_combat_config_layout = gui.Frame(tl('Combat Configurator'), combat_config_layout, title_color=gui_text_color)

    misc_utils_layout = [
        [gui.Text(dev_utils_notice, text_color=gui_text_color)],
        [
            gui.Text(tl('Scale') + ':', text_color=gui_text_color), gui.InputText(size=(8, 1), key='scale'),
            hotkey_button(tl('Set Scale'), GUIKeys.button_set_scale)
        ],
        [gui.Text('Select a pet world:', text_color=gui_text_color), gui.Combo(['WizardCity', 'Krokotopia', 'Marleybone', 'Mooshu', 'Dragonspyre'], default_value='WizardCity', readonly=True,text_color=gui_text_color, size=(13, 1), key='PetWorldInput')], #, hotkey_button('Set Auto Pet World', True)
    ]

    framed_misc_utils_layout = gui.Frame(tl('Misc Utils'), misc_utils_layout, title_color=gui_text_color)

    console_layout = [
        [gui.Text(helpful_support_notice, text_color=gui_text_color)],
        [gui.Multiline(autoscroll=True, horizontal_scroll=True, no_scrollbar=False, key="-CONSOLE-", size=(64, 11), text_color=gui_text_color, disabled = True, echo_stdout_stderr = True)],
        [
            hotkey_button(tl('Collapse / Expand Logs'), GUIKeys.toggle_show_expanded_logs, True),
            hotkey_button(tl('Copy Logs'), GUIKeys.copy_logs, True)
        ],
    ]

    framed_console_layout = gui.Frame(tl('Debug Console'), console_layout, title_color=gui_text_color)

    entity_tp_layout = [
        [gui.Text(dev_utils_notice, text_color=gui_text_color)],
        [gui.Text(tl('Entity teleport uses stored zone data to find and teleport to specific entities.'), text_color=gui_text_color)],
        [
            gui.Text(tl('Entity Name') + ':', text_color=gui_text_color),
            gui.Combo([], size=(30, 1), key='EntityComboInput', readonly=False, enable_events=True),
            hotkey_button(tl('Refresh Entities'), GUIKeys.button_refresh_entities, True)
        ],
        [
            hotkey_button(tl('Go To Entity'), GUIKeys.button_go_to_entity, True),
            hotkey_button(tl('Mass Go To Entity'), GUIKeys.button_mass_go_to_entity, True)
        ],
        [gui.Text(tl('Current Zone') + ': ', key='EntityCurrentZone', text_color=gui_text_color)],
        [gui.Multiline(tl('Available entities will appear here after refreshing.'), key='entity_list_display', size=(64, 15), text_color=gui_text_color, disabled=True, horizontal_scroll=True)]
    ]

    framed_entity_tp_layout = gui.Frame(tl('Entity Teleport'), entity_tp_layout, title_color=gui_text_color)

    tabs = [
        [
            gui.Tab(tl('Hotkeys'), [[framed_toggles_layout, framed_hotkeys_layout, framed_mass_hotkeys_layout, framed_utils_layout]], title_color=gui_text_color),
            gui.Tab(tl('Camera'), [[framed_camera_controls_layout]], title_color=gui_text_color),
            gui.Tab(tl('Dev Utils'), [[framed_custom_tp_layout], [framed_dev_utils_layout]], title_color=gui_text_color),
            gui.Tab(tl('Entity TP'), [[framed_entity_tp_layout]], title_color=gui_text_color),
            gui.Tab(tl('Stat Viewer'), [[framed_stat_viewer_layout]], title_color=gui_text_color),
            gui.Tab(tl('Flythrough'), [[framed_flythrough_layout]], title_color=gui_text_color),
            gui.Tab(tl('Bot'), [[framed_bot_creator_layout]], title_color=gui_text_color),
            gui.Tab(tl('Combat'), [[framed_combat_config_layout]], title_color=gui_text_color),
            gui.Tab(tl('Misc'), [[framed_misc_utils_layout]], title_color=gui_text_color),
            gui.Tab(tl('Console'), [[framed_console_layout]], title_color=gui_text_color),
        ]
    ]

    layout = [
        [gui.Text(tl('Deimos will always be a free tool. If you paid for this, you got scammed!'))],
        [gui.TabGroup(tabs)],
        [client_info_layout]
    ]

    window = gui.Window(title= f'{tool_name} GUI v{tool_version}', layout= layout, keep_on_top=gui_on_top, finalize=True, icon="..\\Deimos-logo.ico", enable_close_attempted_event = False)
    # window.TKroot.iconbitmap(default = "..\Deimos-logo.icon")
    return window

def show_ui_tree_popup(ui_tree_content):
    ui_tree_list = ui_tree_content.splitlines()

    path_dict = {}
    path_stack = []

    for line in ui_tree_list:
        indent = len(line) - len(line.lstrip('-'))
        clean_line = line.lstrip('- ')

        name_match = re.search(r'\[(.*?)\]', clean_line)
        if name_match:
            name = name_match.group(1)
        else:
            name = clean_line.split()[0]  # Fallback to the first word if no brackets

        while len(path_stack) > indent:
            path_stack.pop()

        current_path = path_stack.copy()
        current_path.append(name)

        path_dict[line] = current_path[1:] if len(current_path) > 1 else current_path
        path_stack.append(name)

    layout = [
        [gui.Text('Click the path needed to copy it to clipboard.')],
        [gui.Listbox(values=ui_tree_list, size=(80, 20), key='-TREE-', enable_events=True)],
        [gui.Input(key='-SEARCH-', enable_events=True)],
        [gui.Button('Close')]
    ]
    UITreeWindow = gui.Window('UI Tree', layout, finalize=True, icon="..\\Deimos-logo.ico", keep_on_top=True)

    while True:
        event, values = UITreeWindow.read()
        if event == gui.WINDOW_CLOSED or event == 'Close':
            break
        elif event == '-SEARCH-':
            search_term = values['-SEARCH-'].lower()
            filtered_list = [line for line in ui_tree_list if search_term in line.lower()]
            UITreeWindow['-TREE-'].update(filtered_list)
        elif event == '-TREE-' and values['-TREE-']:
            selected_line = values['-TREE-'][0]
            path = path_dict[selected_line]
            UITreeWindow.close()
            path_str = str(path)
            pyperclip.copy(path_str)
            return path_str

    UITreeWindow.close()

def show_entity_list_popup(entity_list_content):
    entity_list = entity_list_content.splitlines()

    layout = [
        [gui.Text('Click the entity needed to copy the name and location to clipboard.')],
        [gui.Listbox(values=entity_list, size=(80, 20), key='-TREE-', enable_events=True)],
        [gui.Input(key='-SEARCH-', enable_events=True)],
        [gui.Button('Close')]
    ]
    EntityListWindow = gui.Window('Entity List', layout, finalize=True, icon="..\\Deimos-logo.ico", keep_on_top=True)

    while True:
        event, values = EntityListWindow.read()
        print(event)
        if event == gui.WINDOW_CLOSED or event == 'Close':
            break
        elif event == '-SEARCH-':
            search_term = values['-SEARCH-'].lower()
            filtered_list = [line for line in entity_list if search_term in line.lower()]
            EntityListWindow['-TREE-'].update(filtered_list)
        elif event == '-TREE-' and values['-TREE-']:
            selected_line = values['-TREE-'][0]
            EntityListWindow.close()
            pyperclip.copy(selected_line)
            return selected_line

    EntityListWindow.close()

def manage_gui(send_queue: queue.Queue, recv_queue: queue.Queue, gui_theme, gui_text_color, gui_button_color, tool_name, tool_version, gui_on_top, langcode):
    window = create_gui(gui_theme, gui_text_color, gui_button_color, tool_name, tool_version, gui_on_top, langcode)
    global console_sink
    global console_psg
    console_psg = PsgSink(window, '-CONSOLE-')
    console_sink = logger.add(console_psg, colorize=True)

    running = True

    while running:
        event, inputs = window.read(timeout=10)

        # Program commands
        try:
            # Eat as much as the queue gives us. We will be freed by exception
            while True:
                com = recv_queue.get_nowait()
                match com.com_type:
                    case GUICommandType.Close:
                        running = False

                    case GUICommandType.CloseFromBackend:                        
                        event = gui.WINDOW_CLOSE_ATTEMPTED_EVENT
                        # running = False
                        # if not com.data[0]:
                        #     event = gui.WINDOW_CLOSED

                    case GUICommandType.UpdateWindow:
                        window[com.data[0]].update(com.data[1])

                    case GUICommandType.UpdateWindowValues:
                        window[com.data[0]].update(values=com.data[1])

                    case GUICommandType.UpdateConsole:
                        console_psg.toggle_show_expanded_logs()

                    case GUICommandType.ShowUITreePopup:
                        show_ui_tree_popup(com.data)

                    case GUICommandType.ShowEntityListPopup:
                        show_entity_list_popup(com.data)

                    case GUICommandType.CopyConsole:
                        console_psg.copy()

        except queue.Empty:
            pass

        # Window events
        match event:
            case gui.WINDOW_CLOSED:
                running = False
                send_queue.put(GUICommand(GUICommandType.Close))

            case gui.WINDOW_CLOSE_ATTEMPTED_EVENT:
                send_queue.put(GUICommand(GUICommandType.AttemptedClose))

            # Toggles
            case GUIKeys.toggle_speedhack | GUIKeys.toggle_combat | GUIKeys.toggle_dialogue | GUIKeys.toggle_sigil | \
                GUIKeys.toggle_questing | GUIKeys.toggle_auto_pet | GUIKeys.toggle_auto_potion | GUIKeys.toggle_freecam | \
                GUIKeys.toggle_camera_collision | GUIKeys.toggle_show_expanded_logs:
                send_queue.put(GUICommand(GUICommandType.ToggleOption, event))

            # Copying
            case GUIKeys.copy_zone | GUIKeys.copy_position | GUIKeys.copy_rotation | \
                GUIKeys.copy_entity_list | GUIKeys.copy_camera_position | \
                GUIKeys.copy_camera_rotation | GUIKeys.copy_ui_tree | GUIKeys.copy_stats | GUIKeys.copy_logs:
                send_queue.put(GUICommand(GUICommandType.Copy, event))


            # Simple teleports
            case GUIKeys.hotkey_quest_tp | GUIKeys.mass_hotkey_mass_tp | GUIKeys.hotkey_freecam_tp:
                send_queue.put(GUICommand(GUICommandType.Teleport, event))


            # Custom tp
            case GUIKeys.button_custom_tp:
                tp_inputs = [inputs['XInput'], inputs['YInput'], inputs['ZInput'], inputs['YawInput']]
                if any(tp_inputs):
                    send_queue.put(GUICommand(GUICommandType.CustomTeleport, {
                        'X': tp_inputs[0],
                        'Y': tp_inputs[1],
                        'Z': tp_inputs[2],
                        'Yaw': tp_inputs[3],
                    }))

            # Entity tp
            case GUIKeys.button_entity_tp:
                if inputs['EntityTPInput']:
                    send_queue.put(GUICommand(GUICommandType.EntityTeleport, inputs['EntityTPInput']))

            # XYZ Sync
            case GUIKeys.mass_hotkey_xyz_sync:
                send_queue.put(GUICommand(GUICommandType.XYZSync))

            # X Press
            case GUIKeys.mass_hotkey_x_press:
                send_queue.put(GUICommand(GUICommandType.XPress))

            # Cam stuff
            case GUIKeys.button_anchor:
                send_queue.put(GUICommand(GUICommandType.AnchorCam, inputs['CamEntityInput']))

            case GUIKeys.button_set_camera_position:
                camera_inputs = [inputs['CamXInput'], inputs['CamYInput'], inputs['CamZInput'], inputs['CamYawInput'], inputs['CamRollInput'], inputs['CamPitchInput']]
                if any(camera_inputs):
                    send_queue.put(GUICommand(GUICommandType.SetCamPosition, {
                        'X': camera_inputs[0],
                        'Y': camera_inputs[1],
                        'Z': camera_inputs[2],
                        'Yaw': camera_inputs[3],
                        'Roll': camera_inputs[4],
                        'Pitch': camera_inputs[5],
                    }))

            case GUIKeys.button_set_distance:
                distance_inputs = [inputs['CamDistanceInput'], inputs['CamMinInput'], inputs['CamMaxInput']]
                if any(distance_inputs):
                    send_queue.put(GUICommand(GUICommandType.SetCamDistance, {
                        "Distance": distance_inputs[0],
                        "Min": distance_inputs[1],
                        "Max": distance_inputs[2],
                    }))

            # Gotos
            case GUIKeys.button_go_to_zone:
                if inputs['ZoneInput']:
                    send_queue.put(GUICommand(GUICommandType.GoToZone, (False, str(inputs['ZoneInput']))))

            case GUIKeys.button_mass_go_to_zone:
                if inputs['ZoneInput']:
                    send_queue.put(GUICommand(GUICommandType.GoToZone, (True, str(inputs['ZoneInput']))))

            case GUIKeys.button_go_to_world:
                if inputs['WorldInput']:
                    send_queue.put(GUICommand(GUICommandType.GoToWorld, (False, inputs['WorldInput'])))

            case GUIKeys.button_mass_go_to_world:
                if inputs['WorldInput']:
                    send_queue.put(GUICommand(GUICommandType.GoToWorld, (True, inputs['WorldInput'])))

            case GUIKeys.button_go_to_bazaar:
                send_queue.put(GUICommand(GUICommandType.GoToBazaar, False))

            case GUIKeys.button_mass_go_to_bazaar:
                send_queue.put(GUICommand(GUICommandType.GoToBazaar, True))

            case GUIKeys.button_refill_potions:
                send_queue.put(GUICommand(GUICommandType.RefillPotions, False))

            case GUIKeys.button_mass_refill_potions:
                send_queue.put(GUICommand(GUICommandType.RefillPotions, True))

            case GUIKeys.button_execute_flythrough:
                send_queue.put(GUICommand(GUICommandType.ExecuteFlythrough, inputs['flythrough_creator']))

            case GUIKeys.button_kill_flythrough:
                send_queue.put(GUICommand(GUICommandType.KillFlythrough))

            case GUIKeys.button_run_bot:
                send_queue.put(GUICommand(GUICommandType.ExecuteBot, inputs['bot_creator']))

            case GUIKeys.button_set_playstyles:
                send_queue.put(GUICommand(GUICommandType.SetPlaystyles, inputs["combat_config"]))

            case GUIKeys.button_kill_bot:
                send_queue.put(GUICommand(GUICommandType.KillBot))

            case GUIKeys.button_set_scale:
                send_queue.put(GUICommand(GUICommandType.SetScale, inputs['scale']))

            case GUIKeys.button_go_to_entity:
                if inputs['EntityComboInput']:
                    send_queue.put(GUICommand(GUICommandType.GoToEntity, (False, inputs['EntityComboInput'])))

            case GUIKeys.button_mass_go_to_entity:
                if inputs['EntityComboInput']:
                    send_queue.put(GUICommand(GUICommandType.GoToEntity, (True, inputs['EntityComboInput'])))

            case GUIKeys.button_refresh_entities:
                send_queue.put(GUICommand(GUICommandType.RefreshEntities))

            case GUIKeys.button_view_stats:
                enemy_index = re.sub(r'[^0-9]', '', str(inputs['EnemyInput']))
                ally_index = re.sub(r'[^0-9]', '', str(inputs['AllyInput']))
                base_damage = re.sub(r'[^0-9]', '', str(inputs['DamageInput']))
                school_id: int = school_id_to_names[inputs['SchoolInput']]
                send_queue.put(GUICommand(GUICommandType.SelectEnemy, (int(enemy_index), int(ally_index), base_damage, school_id, inputs['CritStatus'], inputs['ForceSchoolStatus'])))

            case GUIKeys.button_swap_members:
                enemy_input = inputs['EnemyInput']
                ally_input = inputs['AllyInput']
                window['EnemyInput'].update(ally_input)
                window['AllyInput'].update(enemy_input)

            # case 'Set Auto Pet World':
            # 	if inputs['PetWorldInput']:
            # 		send_queue.put(GUICommand(GUICommandType.SetPetWorld, (False, str(inputs['PetWorldInput']))))

            # Other
            case _:
                pass

        #Updates pet world when it changes, without the need for a button press -slack
        if inputs and inputs['PetWorldInput'] != wizard_city_dance_game_path[-1]:
            assign_pet_level(inputs['PetWorldInput'])

        def import_check(input_window_str: str, output_window_str: str):
            if inputs and inputs[input_window_str]:
                with open(inputs[input_window_str]) as file:
                    file_data = file.readlines()
                    file_str = ''.join(file_data)
                    window[output_window_str].update(file_str)
                    window[input_window_str].update('')
                    file.close()

        def export_check(path_window_str: str, content_window_str: str):
            if inputs and inputs[path_window_str]:
                file = open(inputs[path_window_str], 'w')
                file.write(inputs[content_window_str])
                file.close()
                window[path_window_str].update('')

        import_check('flythrough_file_path', 'flythrough_creator')
        export_check('flythrough_save_path', 'flythrough_creator')

        import_check('bot_file_path', 'bot_creator')
        export_check('bot_save_path', 'bot_creator')

        import_check('combat_file_path', 'combat_config')
        export_check('combat_save_path', 'combat_config')

    # gui.WIN_CLOSED
    window.close()
    # print("AHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
    # raise GUIClosedException
