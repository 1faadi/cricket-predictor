import os
import json
import pandas as pd
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv
import together
from datetime import datetime
import logging
from typing import Dict, Any, Optional, List
import hashlib
import time
import requests
from collections import defaultdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('psl_parser.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
DATA_FOLDER = "data/psl_json"
POSTGRES_URL = os.getenv("POSTGRES_URL")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
TOGETHERAI_API_KEY = os.getenv("TOGETHERAI_API_KEY")
TOGETHER_EMBEDDING_MODEL = os.getenv("TOGETHER_EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
COLLECTION_NAME = "match_summaries"
BATCH_SIZE = 10  # Process in batches to avoid memory issues

class PSLDataProcessor:
    def __init__(self):
        self.setup_postgres()
        self.setup_qdrant()
        self.processed_count = 0
        self.failed_count = 0
        
    def setup_postgres(self):
        """Setup PostgreSQL connection with better error handling"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(POSTGRES_URL)
            
            self.pg_conn_info = {
                "dbname": parsed.path[1:],
                "user": parsed.username,
                "password": parsed.password,
                "host": parsed.hostname,
                "port": parsed.port or 5432
            }
            
            self.conn = psycopg2.connect(**self.pg_conn_info)
            self.cursor = self.conn.cursor()
            self.create_tables()
            logger.info("PostgreSQL connection established successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup PostgreSQL: {e}")
            raise
    
    def setup_qdrant(self):
        """Setup Qdrant connection"""
        try:
            self.qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            
            # Check if collection exists, create if not
            try:
                self.qdrant.get_collection(COLLECTION_NAME)
                logger.info("Using existing Qdrant collection")
            except:
                self.qdrant.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
                )
                logger.info("Qdrant connection established and collection created")
            
        except Exception as e:
            logger.error(f"Failed to setup Qdrant: {e}")
            raise
    
    def create_tables(self):
        """Create all necessary PostgreSQL tables"""
        
        # Main matches table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS matches (
                match_id TEXT PRIMARY KEY,
                team1 TEXT NOT NULL,
                team2 TEXT NOT NULL,
                match_date DATE,
                venue TEXT,
                city TEXT,
                season TEXT,
                event_name TEXT,
                event_stage TEXT,
                match_type TEXT,
                gender TEXT,
                overs INTEGER,
                balls_per_over INTEGER,
                winner TEXT,
                win_by_runs INTEGER,
                win_by_wickets INTEGER,
                win_method TEXT,
                toss_winner TEXT,
                toss_decision TEXT,
                player_of_match TEXT[],
                umpires TEXT[],
                tv_umpires TEXT[],
                match_referees TEXT[],
                reserve_umpires TEXT[],
                raw_data JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Team squads
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS team_squads (
                id SERIAL PRIMARY KEY,
                match_id TEXT REFERENCES matches(match_id) ON DELETE CASCADE,
                team_name TEXT NOT NULL,
                player_name TEXT NOT NULL,
                player_registry_id TEXT,
                UNIQUE(match_id, team_name, player_name)
            )
        """)
        
        # Innings summary
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS innings_summary (
                id SERIAL PRIMARY KEY,
                match_id TEXT REFERENCES matches(match_id) ON DELETE CASCADE,
                team_name TEXT NOT NULL,
                innings_number INTEGER NOT NULL,
                total_runs INTEGER DEFAULT 0,
                total_wickets INTEGER DEFAULT 0,
                total_overs DECIMAL(4,1) DEFAULT 0,
                total_balls INTEGER DEFAULT 0,
                run_rate DECIMAL(5,2) DEFAULT 0,
                target_runs INTEGER,
                target_overs INTEGER,
                extras JSONB,
                UNIQUE(match_id, innings_number)
            )
        """)
        
        # Player statistics
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS player_stats (
                id SERIAL PRIMARY KEY,
                match_id TEXT REFERENCES matches(match_id) ON DELETE CASCADE,
                team_name TEXT NOT NULL,
                player_name TEXT NOT NULL,
                innings_number INTEGER,
                role TEXT CHECK (role IN ('batter', 'bowler', 'fielder', 'allrounder')),
                runs_scored INTEGER DEFAULT 0,
                balls_faced INTEGER DEFAULT 0,
                fours INTEGER DEFAULT 0,
                sixes INTEGER DEFAULT 0,
                strike_rate DECIMAL(6,2) DEFAULT 0,
                balls_bowled INTEGER DEFAULT 0,
                overs_bowled DECIMAL(4,1) DEFAULT 0,
                runs_conceded INTEGER DEFAULT 0,
                wickets_taken INTEGER DEFAULT 0,
                economy_rate DECIMAL(5,2) DEFAULT 0,
                extras_conceded JSONB,
                dismissal_info JSONB,
                UNIQUE(match_id, player_name, innings_number, role)
            )
        """)
        
        # Create indexes for better performance
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(match_date);",
            "CREATE INDEX IF NOT EXISTS idx_matches_teams ON matches(team1, team2);",
            "CREATE INDEX IF NOT EXISTS idx_matches_season ON matches(season);",
            "CREATE INDEX IF NOT EXISTS idx_player_stats_player ON player_stats(player_name);",
            "CREATE INDEX IF NOT EXISTS idx_player_stats_match ON player_stats(match_id);",
        ]
        
        for index in indexes:
            self.cursor.execute(index)
        
        self.conn.commit()
        logger.info("Database tables created successfully")
    
    def safe_get(self, data: Dict, path: str, default=None):
        """Safely get nested dictionary values"""
        keys = path.split('.')
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current
    
    def validate_json_structure(self, match_json: Dict) -> bool:
        """Validate that JSON has required structure"""
        required_fields = ['info']
        
        for field in required_fields:
            if field not in match_json:
                logger.warning(f"Missing required field: {field}")
                return False
        
        # Check if info has teams
        if not self.safe_get(match_json, 'info.teams'):
            logger.warning("Missing teams information")
            return False
            
        return True
    
    def parse_date(self, date_str: str) -> Optional[datetime.date]:
        """Parse date string with multiple format support"""
        if not date_str:
            return None
            
        date_formats = [
            "%Y-%m-%d",
            "%d/%m/%Y", 
            "%m/%d/%Y",
            "%Y/%m/%d",
            "%d-%m-%Y",
            "%m-%d-%Y"
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(str(date_str), fmt).date()
            except (ValueError, TypeError):
                continue
        
        logger.warning(f"Could not parse date: {date_str}")
        return None
    
    def calculate_innings_stats(self, innings_data: Dict) -> Dict:
        """Calculate comprehensive innings statistics"""
        stats = {
            'total_runs': 0,
            'total_wickets': 0,
            'total_balls': 0,
            'total_overs': 0.0,
            'run_rate': 0.0,
            'extras': defaultdict(int)
        }
        
        try:
            overs = innings_data.get('overs', [])
            
            for over_data in overs:
                deliveries = over_data.get('deliveries', [])
                
                for delivery in deliveries:
                    # Count balls (excluding wides and no-balls in some formats)
                    stats['total_balls'] += 1
                    
                    # Runs calculation
                    runs = delivery.get('runs', {})
                    stats['total_runs'] += runs.get('total', 0)
                    
                    # Extras tracking
                    extras = delivery.get('extras', {})
                    for extra_type, extra_runs in extras.items():
                        stats['extras'][extra_type] += extra_runs
                    
                    # Wickets
                    wickets = delivery.get('wickets', [])
                    stats['total_wickets'] += len(wickets)
            
            # Calculate overs
            complete_overs = stats['total_balls'] // 6
            remaining_balls = stats['total_balls'] % 6
            stats['total_overs'] = complete_overs + (remaining_balls / 10.0) if remaining_balls > 0 else complete_overs
            
            # Calculate run rate
            if stats['total_overs'] > 0:
                stats['run_rate'] = round(stats['total_runs'] / stats['total_overs'], 2)
            
        except Exception as e:
            logger.error(f"Error calculating innings stats: {e}")
        
        return stats
    
    def extract_player_stats(self, match_json: Dict, match_id: str):
        """Extract detailed player statistics"""
        player_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        player_roles = defaultdict(set)
        
        try:
            innings_list = match_json.get('innings', [])
            
            for innings_num, innings in enumerate(innings_list, 1):
                team = innings.get('team', 'Unknown')
                overs = innings.get('overs', [])
                
                for over_data in overs:
                    for delivery in over_data.get('deliveries', []):
                        # Batting stats
                        batter = delivery.get('batter')
                        if batter:
                            runs = delivery.get('runs', {})
                            batter_runs = runs.get('batter', 0)
                            
                            stats = player_stats[batter][innings_num]
                            stats['runs_scored'] += batter_runs
                            stats['balls_faced'] += 1
                            stats['team_name'] = team
                            
                            if batter_runs == 4:
                                stats['fours'] += 1
                            elif batter_runs == 6:
                                stats['sixes'] += 1
                            
                            player_roles[batter].add('batter')
                        
                        # Bowling stats
                        bowler = delivery.get('bowler')
                        if bowler:
                            runs = delivery.get('runs', {})
                            total_runs = runs.get('total', 0)
                            
                            # Determine bowling team (opposite of batting team)
                            bowling_teams = self.safe_get(match_json, 'info.teams', [])
                            bowling_team = next((t for t in bowling_teams if t != team), 'Unknown')
                            
                            stats = player_stats[bowler][innings_num]
                            stats['balls_bowled'] += 1
                            stats['runs_conceded'] += total_runs
                            stats['team_name'] = bowling_team
                            
                            # Track extras
                            extras = delivery.get('extras', {})
                            if not stats.get('extras_conceded'):
                                stats['extras_conceded'] = defaultdict(int)
                            for extra_type, extra_runs in extras.items():
                                stats['extras_conceded'][extra_type] += extra_runs
                            
                            # Wickets
                            wickets = delivery.get('wickets', [])
                            stats['wickets_taken'] += len(wickets)
                            
                            player_roles[bowler].add('bowler')
        
        except Exception as e:
            logger.error(f"Error extracting player stats for match {match_id}: {e}")
        
        return player_stats, player_roles
    
    def determine_player_role(self, roles: set, stats: Dict) -> str:
        """Determine primary player role based on participation"""
        if 'batter' in roles and 'bowler' in roles:
            return 'allrounder'
        elif 'bowler' in roles:
            return 'bowler'
        elif 'batter' in roles:
            return 'batter'
        else:
            return 'fielder'
    
    def create_match_summary(self, match_json: Dict) -> str:
        """Create comprehensive match summary for vector embedding"""
        try:
            info = match_json.get('info', {})
            teams = info.get('teams', ['Team A', 'Team B'])
            
            # Basic match info
            summary_parts = []
            
            # Event and teams
            event = info.get('event', {})
            event_name = event.get('name', 'Cricket Match')
            event_stage = event.get('stage', '')
            
            if event_stage:
                summary_parts.append(f"{event_name} {event_stage} between {teams[0]} and {teams[1]}")
            else:
                summary_parts.append(f"{event_name} between {teams[0]} and {teams[1]}")
            
            # Date and venue
            dates = info.get('dates', [])
            if dates:
                summary_parts.append(f"played on {dates[0]}")
            
            venue = info.get('venue', '')
            city = self.safe_get(info, 'city')
            if venue:
                location = f"at {venue}"
                if city:
                    location += f" in {city}"
                summary_parts.append(location)
            
            # Season
            season = info.get('season', '')
            if season:
                summary_parts.append(f"during {season} season")
            
            # Toss
            toss = info.get('toss', {})
            if toss.get('winner') and toss.get('decision'):
                summary_parts.append(f"{toss['winner']} won toss and elected to {toss['decision']}")
            
            # Match outcome
            outcome = info.get('outcome', {})
            if outcome.get('winner'):
                win_details = ""
                if 'by' in outcome:
                    by_data = outcome['by']
                    if 'runs' in by_data:
                        win_details = f" by {by_data['runs']} runs"
                    elif 'wickets' in by_data:
                        win_details = f" by {by_data['wickets']} wickets"
                summary_parts.append(f"{outcome['winner']} won{win_details}")
            elif outcome.get('result'):
                summary_parts.append(f"Result: {outcome['result']}")
            
            # Player of match
            player_of_match = info.get('player_of_match', [])
            if player_of_match:
                if len(player_of_match) == 1:
                    summary_parts.append(f"{player_of_match[0]} was Player of the Match")
                else:
                    summary_parts.append(f"Players of Match: {', '.join(player_of_match)}")
            
            # Innings summaries
            innings = match_json.get('innings', [])
            for i, innings_data in enumerate(innings):
                team = innings_data.get('team', f'Team {i+1}')
                stats = self.calculate_innings_stats(innings_data)
                summary_parts.append(
                    f"{team} scored {stats['total_runs']}/{stats['total_wickets']} "
                    f"in {stats['total_overs']} overs (RR: {stats['run_rate']})"
                )
            
            return '. '.join(summary_parts) + '.'
            
        except Exception as e:
            logger.error(f"Error creating match summary: {e}")
            return f"Cricket match between {teams[0] if teams else 'Team A'} and {teams[1] if len(teams) > 1 else 'Team B'}"
    
    def get_embedding_with_retry(self, text: str, max_retries: int = 3) -> Optional[List[float]]:
        """Get embedding using direct HTTP request to TogetherAI API"""
        url = "https://api.together.xyz/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {TOGETHERAI_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": TOGETHER_EMBEDDING_MODEL,
            "input": [text]
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                return response.json()["data"][0]["embedding"]
            except Exception as e:
                logger.warning(f"Embedding attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

        logger.error(f"Failed to get embedding after {max_retries} attempts")
        return None

    def insert_match_data(self, match_id: str, match_json: Dict):
        """Insert match data into PostgreSQL"""
        try:
            info = match_json.get('info', {})
            teams = info.get('teams', ['Team A', 'Team B'])
            
            # Parse basic match info
            dates = info.get('dates', [])
            match_date = self.parse_date(dates[0]) if dates else None
            
            venue = info.get('venue', '')
            city = self.safe_get(info, 'city', '')
            season = info.get('season', '')
            
            event = info.get('event', {})
            event_name = event.get('name', '')
            event_stage = event.get('stage', '')
            
            match_type = info.get('match_type', '')
            gender = info.get('gender', '')
            overs = info.get('overs')
            balls_per_over = info.get('balls_per_over', 6)
            
            # Outcome information
            outcome = info.get('outcome', {})
            winner = outcome.get('winner')
            win_by_runs = None
            win_by_wickets = None
            win_method = None
            
            if 'by' in outcome:
                by_data = outcome['by']
                if 'runs' in by_data:
                    win_by_runs = by_data['runs']
                    win_method = 'runs'
                elif 'wickets' in by_data:
                    win_by_wickets = by_data['wickets']
                    win_method = 'wickets'
            
            # Toss and officials
            toss = info.get('toss', {})
            officials = info.get('officials', {})
            player_of_match = info.get('player_of_match', [])
            
            # Insert main match record
            self.cursor.execute("""
                INSERT INTO matches (
                    match_id, team1, team2, match_date, venue, city, season, 
                    event_name, event_stage, match_type, gender, overs, balls_per_over,
                    winner, win_by_runs, win_by_wickets, win_method,
                    toss_winner, toss_decision, player_of_match,
                    umpires, tv_umpires, match_referees, reserve_umpires, raw_data
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (match_id) DO UPDATE SET
                    raw_data = EXCLUDED.raw_data,
                    created_at = CURRENT_TIMESTAMP
            """, (
                match_id, teams[0], teams[1], match_date, venue, city, season,
                event_name, event_stage, match_type, gender, overs, balls_per_over,
                winner, win_by_runs, win_by_wickets, win_method,
                toss.get('winner'), toss.get('decision'), player_of_match,
                officials.get('umpires', []), officials.get('tv_umpires', []),
                officials.get('match_referees', []), officials.get('reserve_umpires', []),
                Json(match_json)  # Store full JSON for reference
            ))
            
            # Insert team squads
            players_data = info.get('players', {})
            registry = self.safe_get(info, 'registry.people', {})
            
            for team_name, player_list in players_data.items():
                for player_name in player_list:
                    player_id = registry.get(player_name)
                    self.cursor.execute("""
                        INSERT INTO team_squads (match_id, team_name, player_name, player_registry_id)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (match_id, team_name, player_name) DO NOTHING
                    """, (match_id, team_name, player_name, player_id))
            
            # Insert innings summaries
            innings = match_json.get('innings', [])
            for i, innings_data in enumerate(innings):
                team_name = innings_data.get('team', f'Team {i+1}')
                stats = self.calculate_innings_stats(innings_data)
                target = innings_data.get('target', {})
                
                self.cursor.execute("""
                    INSERT INTO innings_summary (
                        match_id, team_name, innings_number, total_runs, total_wickets,
                        total_overs, total_balls, run_rate, target_runs, target_overs, extras
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (match_id, innings_number) DO UPDATE SET
                        total_runs = EXCLUDED.total_runs,
                        total_wickets = EXCLUDED.total_wickets,
                        total_overs = EXCLUDED.total_overs,
                        run_rate = EXCLUDED.run_rate
                """, (
                    match_id, team_name, i + 1, stats['total_runs'], stats['total_wickets'],
                    stats['total_overs'], stats['total_balls'], stats['run_rate'],
                    target.get('runs'), target.get('overs'), Json(dict(stats['extras']))
                ))
            
            # Insert player statistics
            player_stats, player_roles = self.extract_player_stats(match_json, match_id)
            
            for player_name, innings_stats in player_stats.items():
                for innings_num, stats in innings_stats.items():
                    role = self.determine_player_role(player_roles[player_name], stats)
                    
                    # Calculate derived stats
                    strike_rate = (stats['runs_scored'] / stats['balls_faced'] * 100) if stats['balls_faced'] > 0 else 0
                    economy_rate = (stats['runs_conceded'] / (stats['balls_bowled'] / 6)) if stats['balls_bowled'] > 0 else 0
                    overs_bowled = stats['balls_bowled'] / 6 if stats['balls_bowled'] > 0 else 0
                    
                    self.cursor.execute("""
                        INSERT INTO player_stats (
                            match_id, team_name, player_name, innings_number, role,
                            runs_scored, balls_faced, fours, sixes, strike_rate,
                            balls_bowled, overs_bowled, runs_conceded, wickets_taken, economy_rate,
                            extras_conceded
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (match_id, player_name, innings_number, role) DO UPDATE SET
                            runs_scored = EXCLUDED.runs_scored,
                            balls_faced = EXCLUDED.balls_faced,
                            strike_rate = EXCLUDED.strike_rate,
                            wickets_taken = EXCLUDED.wickets_taken
                    """, (
                        match_id, stats['team_name'], player_name, innings_num, role,
                        stats['runs_scored'], stats['balls_faced'], stats['fours'], stats['sixes'], round(strike_rate, 2),
                        stats['balls_bowled'], round(overs_bowled, 1), stats['runs_conceded'], stats['wickets_taken'], round(economy_rate, 2),
                        Json(dict(stats.get('extras_conceded', {})))
                    ))
            
        except Exception as e:
            logger.error(f"Error inserting match data for {match_id}: {e}")
            raise
    
    def insert_to_qdrant(self, match_id: str, summary: str):
        """Insert match summary vector to Qdrant"""
        try:
            vector = self.get_embedding_with_retry(summary)
            if not vector:
                logger.error(f"Failed to get embedding for match {match_id}")
                return False
            
            # Create a more deterministic ID
            point_id = int(hashlib.md5(match_id.encode()).hexdigest()[:8], 16)
            
            point = PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "match_id": str(match_id),
                    "summary": summary[:1000],  # Limit summary length
                    "created_at": datetime.now().isoformat()
                }
            )
            
            self.qdrant.upsert(collection_name=COLLECTION_NAME, points=[point])
            return True
            
        except Exception as e:
            logger.error(f"Error inserting to Qdrant for match {match_id}: {e}")
            return False
    
    def process_match_file(self, file_path: str, match_id: str) -> bool:
        """Process a single match file"""
        try:
            logger.info(f"Processing match: {match_id}")
            
            # Load and validate JSON
            with open(file_path, 'r', encoding='utf-8') as f:
                match_json = json.load(f)
            
            if not self.validate_json_structure(match_json):
                logger.warning(f"Invalid JSON structure for {match_id}")
                return False
            
            # Create summary
            summary = self.create_match_summary(match_json)
            
            # Insert to databases
            self.insert_match_data(match_id, match_json)
            qdrant_success = self.insert_to_qdrant(match_id, summary)
            
            if qdrant_success:
                self.conn.commit()
                logger.info(f"Successfully processed match: {match_id}")
                return True
            else:
                self.conn.rollback()
                logger.error(f"Failed to process match: {match_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error processing match {match_id}: {e}")
            self.conn.rollback()
            return False
    
    def process_all_matches(self):
        """Process all match files in the data folder"""
        if not os.path.exists(DATA_FOLDER):
            logger.error(f"Data folder '{DATA_FOLDER}' not found!")
            return
        
        # Get all JSON files
        files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.json')]
        
        if not files:
            logger.error("No JSON files found!")
            return
        
        logger.info(f"Found {len(files)} match files to process")
        
        # Process files with progress bar
        for filename in tqdm(files, desc="Processing matches"):
            match_id = filename.replace('.json', '')
            file_path = os.path.join(DATA_FOLDER, filename)
            
            success = self.process_match_file(file_path, match_id)
            
            if success:
                self.processed_count += 1
            else:
                self.failed_count += 1
        
        # Final summary
        logger.info(f"Processing complete!")
        logger.info(f"Successfully processed: {self.processed_count}")
        logger.info(f"Failed: {self.failed_count}")
        logger.info(f"Total files: {len(files)}")
    
    def close_connections(self):
        """Close database connections"""
        try:
            if hasattr(self, 'cursor'):
                self.cursor.close()
            if hasattr(self, 'conn'):
                self.conn.close()
            logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Error closing connections: {e}")

# Main execution
def main():
    processor = PSLDataProcessor()
    
    try:
        processor.process_all_matches()
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        processor.close_connections()

if __name__ == "__main__":
    main()