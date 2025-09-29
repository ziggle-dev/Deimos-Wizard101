import sqlite3
import asyncio
from typing import Optional, List, Tuple
from wizwalker import XYZ, Client
from pathlib import Path
from loguru import logger


class QuestDatabase:
    """Database for tracking quest teleport locations and zone entities."""

    def __init__(self, db_path: str = "quest_data.db"):
        """Initialize the database connection and create tables if needed."""
        self.db_path = db_path
        self.conn = None
        self._init_db()

    def _init_db(self):
        """Create database tables if they don't exist."""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()

        # Table for storing last valid quest TP locations per zone
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS quest_locations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                zone_name TEXT NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                z REAL NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(zone_name)
            )
        """)

        # Table for storing entities per zone
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS zone_entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                zone_name TEXT NOT NULL,
                entity_name TEXT NOT NULL,
                entity_template TEXT,
                behaviors TEXT,
                x REAL NOT NULL,
                y REAL NOT NULL,
                z REAL NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(zone_name, entity_name, x, y, z)
            )
        """)

        # Migrate old databases: add behaviors column if it doesn't exist
        try:
            cursor.execute("SELECT behaviors FROM zone_entities LIMIT 1")
        except sqlite3.OperationalError:
            # Column doesn't exist, add it
            logger.info("Migrating database: adding behaviors column")
            cursor.execute("ALTER TABLE zone_entities ADD COLUMN behaviors TEXT DEFAULT ''")
            self.conn.commit()

        # Index for faster zone lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_zone_name
            ON zone_entities(zone_name)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_entity_name
            ON zone_entities(entity_name)
        """)

        self.conn.commit()

    def store_quest_location(self, zone_name: str, xyz: XYZ):
        """Store or update the last valid quest TP location for a zone."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO quest_locations
                (zone_name, x, y, z, timestamp)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (zone_name, xyz.x, xyz.y, xyz.z))
            self.conn.commit()
            logger.debug(f"Stored quest location for zone {zone_name}: {xyz}")
        except Exception as e:
            logger.error(f"Failed to store quest location: {e}")

    def get_quest_location(self, zone_name: str) -> Optional[XYZ]:
        """Retrieve the last valid quest TP location for a zone."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT x, y, z FROM quest_locations
                WHERE zone_name = ?
            """, (zone_name,))
            result = cursor.fetchone()
            if result:
                xyz = XYZ(x=result[0], y=result[1], z=result[2])
                logger.debug(f"Retrieved quest location for zone {zone_name}: {xyz}")
                return xyz
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve quest location: {e}")
            return None

    def clear_zone_entities(self, zone_name: str):
        """Clear all stored entities for a specific zone."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                DELETE FROM zone_entities WHERE zone_name = ?
            """, (zone_name,))
            self.conn.commit()
            logger.debug(f"Cleared entities for zone {zone_name}")
        except Exception as e:
            logger.error(f"Failed to clear zone entities: {e}")

    def store_zone_entity(self, zone_name: str, entity_name: str, xyz: XYZ, entity_template: str = "", behaviors: str = ""):
        """Store an entity's location in a zone."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO zone_entities
                (zone_name, entity_name, entity_template, behaviors, x, y, z, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (zone_name, entity_name, entity_template, behaviors, xyz.x, xyz.y, xyz.z))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Failed to store entity {entity_name} in zone {zone_name}: {e}")

    def get_zone_entities(self, zone_name: str) -> List[Tuple[str, XYZ, str, str]]:
        """Get all entities stored for a specific zone.

        Returns:
            List of tuples: (entity_name, XYZ, entity_template, behaviors)
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT entity_name, x, y, z, entity_template, behaviors
                FROM zone_entities
                WHERE zone_name = ?
                ORDER BY entity_name
            """, (zone_name,))
            results = cursor.fetchall()
            entities = [(row[0], XYZ(x=row[1], y=row[2], z=row[3]), row[4], row[5] or "") for row in results]
            return entities
        except Exception as e:
            logger.error(f"Failed to retrieve zone entities: {e}")
            return []

    def find_entity_by_name(self, zone_name: str, entity_name: str) -> Optional[Tuple[XYZ, str]]:
        """Find an entity by name in a specific zone.

        Returns:
            Tuple of (XYZ, entity_template) or None if not found
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT x, y, z, entity_template
                FROM zone_entities
                WHERE zone_name = ? AND entity_name LIKE ?
                LIMIT 1
            """, (zone_name, f"%{entity_name}%"))
            result = cursor.fetchone()
            if result:
                return (XYZ(x=result[0], y=result[1], z=result[2]), result[3])
            return None
        except Exception as e:
            logger.error(f"Failed to find entity {entity_name}: {e}")
            return None

    def get_all_entity_names(self, zone_name: str) -> List[str]:
        """Get all unique entity names for a zone."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT DISTINCT entity_name
                FROM zone_entities
                WHERE zone_name = ?
                ORDER BY entity_name
            """, (zone_name,))
            return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to retrieve entity names: {e}")
            return []

    def get_all_entities_with_metadata(self, current_zone: str = None) -> List[str]:
        """Get all entities from database with display names and zone info.

        Returns formatted list like: "Cat Tail [Reagent] (WizardCity)" or "Cat Tail [Reagent] *HERE*"
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT DISTINCT zone_name, entity_name, entity_template, behaviors
                FROM zone_entities
                ORDER BY entity_name
            """)
            results = cursor.fetchall()

            formatted_entities = []
            seen = set()  # Track unique display names to avoid duplicates

            for zone_name, entity_name, template_info, behaviors in results:
                display_name = self.get_display_name(template_info)
                has_display_name = display_name is not None and display_name.strip() != ''

                # Classify the entity
                category = self.classify_entity_by_behavior(behaviors, has_display_name)

                # Convert to readable category
                category_display = {
                    'reagent': 'Reagent',
                    'npc': 'NPC',
                    'quest_object': 'Quest',
                    'wisp': 'Wisp',
                    'duel_circle': 'Duel Circle',
                    'other': 'Other'
                }.get(category, 'Other')

                # Use display name if available, otherwise entity name
                name = display_name if display_name else entity_name

                # Shorten zone name for display
                zone_short = zone_name.split('/')[-1] if '/' in zone_name else zone_name

                # Create unique key
                unique_key = f"{name}|{category}"

                # Format: "Display Name [Category] (Zone)" or "Display Name [Category] *HERE*"
                if current_zone and zone_name == current_zone:
                    formatted = f"{name} [{category_display}] *HERE*"
                else:
                    formatted = f"{name} [{category_display}] ({zone_short})"

                # Only add if we haven't seen this name/category combo
                if unique_key not in seen:
                    formatted_entities.append(formatted)
                    seen.add(unique_key)

            return sorted(formatted_entities)
        except Exception as e:
            logger.error(f"Failed to retrieve all entities: {e}")
            return []

    def parse_entity_selection(self, selection: str) -> Optional[str]:
        """Parse user selection to extract entity name.

        Input: "Cat Tail [Reagent] (WC_Streets)" or "Cat Tail [Reagent] *HERE*"
        Output: "Cat Tail"
        """
        if '[' in selection:
            return selection.split('[')[0].strip()
        return selection.strip()

    def search_entity_across_zones(self, search_term: str) -> List[Tuple[str, str, XYZ]]:
        """Search for entities across all zones.

        Returns:
            List of tuples: (zone_name, entity_name, XYZ)
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT zone_name, entity_name, x, y, z, entity_template
                FROM zone_entities
                WHERE entity_name LIKE ? OR entity_template LIKE ?
                ORDER BY zone_name, entity_name
                LIMIT 100
            """, (f"%{search_term}%", f"%{search_term}%"))
            results = cursor.fetchall()
            return [(row[0], row[1], XYZ(x=row[2], y=row[3], z=row[4])) for row in results]
        except Exception as e:
            logger.error(f"Failed to search entities: {e}")
            return []

    def get_entity_with_zone(self, entity_name: str) -> Optional[Tuple[str, XYZ, str]]:
        """Find entity by name or display name and return its zone, location, and display name.

        Returns:
            Tuple of (zone_name, XYZ, display_name) or None
        """
        try:
            cursor = self.conn.cursor()
            # Search by entity_name or display name in template
            cursor.execute("""
                SELECT zone_name, x, y, z, entity_template
                FROM zone_entities
                WHERE entity_name LIKE ? OR entity_template LIKE ?
                LIMIT 1
            """, (f"%{entity_name}%", f"%{entity_name}%"))
            result = cursor.fetchone()
            if result:
                display_name = self.get_display_name(result[4]) or entity_name
                return (result[0], XYZ(x=result[1], y=result[2], z=result[3]), display_name)
            return None
        except Exception as e:
            logger.error(f"Failed to get entity with zone: {e}")
            return None

    def get_reagents_and_pickups(self, zone_name: str) -> List[Tuple[str, XYZ, str]]:
        """Get entities that are likely reagents or pickups (have display names).

        Returns:
            List of tuples: (entity_name, XYZ, display_name)
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT entity_name, x, y, z, entity_template
                FROM zone_entities
                WHERE zone_name = ? AND entity_template LIKE '%|%'
                ORDER BY entity_name
            """, (zone_name,))
            results = cursor.fetchall()

            # Patterns to exclude (NPCs, mobs, chests, ambient entities)
            exclude_patterns = [
                'StandIn', 'standin',  # NPCs use StandIn naming
                '_NPC_', '_Npc_', 'NPC_',  # Direct NPC naming
                'Ambient', 'ambient',  # Ambient creatures
                'Chest', 'chest',  # Chests (quest/loot containers)
                '_Enemy_', '_Monster_', '_Mob_',  # Enemy creatures
                'Hunter', 'Warrior', 'Wizard', 'Mage',  # Common mob types
                'Guard', 'Soldier', 'Knight',  # Guard type NPCs
                'Ghoulture', 'Ghulture',  # Specific mobs
            ]

            entities = []
            for row in results:
                entity_name = row[0]

                # Check if entity matches exclusion patterns
                is_excluded = any(pattern in entity_name for pattern in exclude_patterns)

                if not is_excluded:
                    # Extract display name from template_info
                    if '|' in row[4]:
                        _, display_name = row[4].split('|', 1)
                        entities.append((row[0], XYZ(x=row[1], y=row[2], z=row[3]), display_name))

            return entities
        except Exception as e:
            logger.error(f"Failed to retrieve reagents/pickups: {e}")
            return []

    def is_reagent_or_pickup(self, entity_template: str) -> bool:
        """Check if an entity is likely a reagent or pickup based on its template info."""
        return '|' in entity_template if entity_template else False

    def get_display_name(self, entity_template: str) -> Optional[str]:
        """Extract display name from entity template info."""
        if self.is_reagent_or_pickup(entity_template):
            parts = entity_template.split('|', 1)
            if len(parts) == 2:
                return parts[1]
        return None

    def is_likely_reagent(self, entity_name: str, display_name: str) -> bool:
        """More intelligent check if entity is actually a reagent/pickup.

        Uses naming patterns and exclusions to filter out NPCs/mobs.
        """
        # Exclude patterns (NPCs, mobs, chests)
        exclude_patterns = [
            'StandIn', 'standin',
            '_NPC_', '_Npc_', 'NPC_',
            'Ambient', 'ambient',
            'Chest', 'chest', 'Crate', 'crate',
            '_Enemy_', '_Monster_', '_Mob_',
            'Hunter', 'Warrior', 'Wizard', 'Mage',
            'Guard', 'Soldier', 'Knight',
            'Ghoulture', 'Ghulture',
            'Dromel', 'PakMan',  # Mirage specific NPCs
        ]

        if any(pattern in entity_name for pattern in exclude_patterns):
            return False

        # Include patterns (common reagent naming)
        include_patterns = [
            'ORE_', 'Ore_', 'ore_',  # Ore reagents
            'STONE_', 'Stone_', 'stone_',  # Stone blocks
            'WOOD_', 'Wood_', 'wood_',  # Wood reagents
            'PLANT_', 'Plant_', 'plant_',  # Plant reagents
            'FLOWER_', 'Flower_', 'flower_',  # Flower reagents
            'MUSHROOM_', 'Mushroom_', 'mushroom_',  # Mushroom reagents
            'MOSS_', 'Moss_', 'moss_',  # Moss reagents
            'CATTAIL', 'Cattail', 'cattail',  # Cat tail
            'MISTWOOD', 'Mistwood', 'mistwood',  # Mist wood
            'SCRAP', 'Scrap', 'scrap',  # Scrap iron
            'PARCHMENT', 'Parchment', 'parchment',
            'SPIDER', 'Spider', 'spider',
            'LOTUS', 'Lotus', 'lotus',
            'NIGHTSHADE', 'Nightshade', 'nightshade',
        ]

        if any(pattern in entity_name for pattern in include_patterns):
            return True

        # If entity name suggests it's a reagent location/holder
        if any(x in entity_name.lower() for x in ['reagent', 'pickup', 'collect']):
            return True

        # Default to False for safety (user can still see in ALL ENTITIES)
        return False

    def classify_entity_by_behavior(self, behaviors: str, has_display_name: bool) -> str:
        """Classify entity type based on its behaviors - UNIVERSAL SOLUTION.

        Returns: 'npc', 'mob', 'reagent', 'other'
        """
        if not behaviors:
            # No behaviors - likely ambient/positional objects or reagents
            return 'reagent' if has_display_name else 'other'

        behavior_list = behaviors.split(',')

        # NPCs have NPCBehavior without being enemies
        if 'NPCBehavior' in behavior_list:
            # Check if it's actually a mob (has NPCBehavior but is_enemy flag)
            # This is checked in sprinty_client.get_mobs() method
            # For now, treat as NPC - mobs will be reclassified if needed
            return 'npc'

        # Collectibles/Reagents typically have these behaviors
        if any(b in behavior_list for b in ['CollectBehavior', 'ReagentBehavior', 'PickupBehavior']):
            return 'reagent'

        # Quest objects
        if 'QuestBehavior' in behavior_list or 'QuestObjectBehavior' in behavior_list:
            return 'quest_object'

        # Wisps
        if 'WispBehavior' in behavior_list:
            return 'wisp'

        # Duel circles/sigils
        if 'DuelCircleBehavior' in behavior_list or 'SigilBehavior' in behavior_list:
            return 'duel_circle'

        # Has display name but no specific behavior - likely a reagent
        if has_display_name:
            return 'reagent'

        # Default
        return 'other'

    def get_entities_by_category(self, zone_name: str) -> dict:
        """Categorize entities using behavior-based classification - UNIVERSAL SOLUTION.

        Returns:
            dict with keys: 'reagents', 'npcs', 'mobs', 'quest_objects', 'wisps', 'duel_circles', 'other'
        """
        all_entities = self.get_zone_entities(zone_name)

        categories = {
            'reagents': [],
            'npcs': [],
            'mobs': [],
            'quest_objects': [],
            'wisps': [],
            'duel_circles': [],
            'other': []
        }

        for entity_name, xyz, template_info, behaviors in all_entities:
            display_name = self.get_display_name(template_info)
            has_display_name = display_name is not None and display_name.strip() != ''

            # Classify based on behaviors
            category = self.classify_entity_by_behavior(behaviors, has_display_name)

            # Convert singular to plural for dict key
            category_key = category + 's' if category in ['reagent', 'npc', 'mob', 'wisp'] else category

            # Add to appropriate category (with fallback to 'other' if key doesn't exist)
            if category_key in categories:
                categories[category_key].append((entity_name, xyz, display_name or '', behaviors))
            else:
                categories['other'].append((entity_name, xyz, display_name or '', behaviors))

        return categories

    async def track_zone_entities(self, client: Client):
        """Scan and store all entities in the current zone."""
        try:
            from src.sprinty_client import SprintyClient
            from wizwalker import MemoryReadError

            zone_name = await client.zone_name()
            logger.debug(f"Tracking entities in zone: {zone_name}")

            # Clear existing entities for this zone to get fresh data
            self.clear_zone_entities(zone_name)

            sprinter = SprintyClient(client)
            entities = await sprinter.get_base_entity_list()

            entity_count = 0
            for entity in entities:
                try:
                    entity_name = await entity.object_name()
                    entity_location = await entity.location()

                    # Get template name and display name
                    template_info = ""
                    try:
                        template = await entity.object_template()
                        template_name = str(await template.object_name())

                        # Try to get display name
                        try:
                            display_code = await template.display_name()
                            if display_code:
                                display_name = await client.cache_handler.get_langcode_name(display_code)
                                if display_name and display_name.strip():
                                    template_info = f"{template_name}|{display_name}"
                                else:
                                    template_info = template_name
                            else:
                                template_info = template_name
                        except:
                            template_info = template_name
                    except:
                        template_info = ""

                    # Get behaviors - THIS IS THE KEY TO CLASSIFICATION
                    behavior_list = []
                    try:
                        behaviors = await entity.inactive_behaviors()
                        for b in behaviors:
                            try:
                                behavior_name = await b.read_type_name()
                                behavior_list.append(behavior_name)
                            except:
                                pass
                    except (ValueError, MemoryReadError):
                        pass

                    # Store behaviors as comma-separated string
                    behaviors_str = ",".join(behavior_list) if behavior_list else ""

                    self.store_zone_entity(zone_name, entity_name, entity_location, template_info, behaviors_str)
                    entity_count += 1
                except Exception as e:
                    # Skip entities that can't be read
                    pass

            logger.debug(f"Stored {entity_count} entities for zone {zone_name}")

        except Exception as e:
            logger.error(f"Failed to track zone entities: {e}")

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            logger.debug("Database connection closed")


# Global database instance
_db_instance: Optional[QuestDatabase] = None


def get_quest_db() -> QuestDatabase:
    """Get or create the global quest database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = QuestDatabase()
    return _db_instance