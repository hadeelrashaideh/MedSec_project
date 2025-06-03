from django.core.management.base import BaseCommand
from django.core.cache import cache
from django.contrib.auth import get_user_model
import os
import base64
import json
import time

User = get_user_model()

class Command(BaseCommand):
    help = 'Exports or imports encryption keys to/from a JSON file for persistence across restarts'

    def add_arguments(self, parser):
        parser.add_argument(
            '--export',
            action='store_true',
            help='Export keys from cache to file',
        )
        
        parser.add_argument(
            '--import',
            action='store_true',
            help='Import keys from file to cache',
        )
        
        parser.add_argument(
            '--file',
            type=str,
            default='encryption_keys.json',
            help='Path to the keys file',
        )
        
        parser.add_argument(
            '--list',
            action='store_true',
            help='List all encryption keys in cache',
        )

    def handle(self, *args, **options):
        if options['export']:
            self.export_keys(options['file'])
        elif options['import']:
            self.import_keys(options['file'])
        elif options['list']:
            self.list_keys()
        else:
            self.stdout.write(self.style.ERROR('Please specify either --export, --import, or --list'))
    
    def export_keys(self, filename):
        """Export all encryption keys from cache to a file"""
        self.stdout.write(f'Exporting encryption keys to {filename}...')
        
        # Get all users
        users = User.objects.all()
        
        # Collect keys for each user
        keys = {}
        for user in users:
            cache_key = f"encryption_key_{user.id}"
            key_value = cache.get(cache_key)
            
            if key_value:
                keys[str(user.id)] = key_value
                self.stdout.write(f'  Found key for user {user.username} (ID: {user.id})')
        
        # Add export timestamp
        export_data = {
            'timestamp': time.time(),
            'keys': keys
        }
        
        # Write to file
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.stdout.write(self.style.SUCCESS(f'Exported {len(keys)} encryption keys to {filename}'))
    
    def import_keys(self, filename):
        """Import encryption keys from file to cache"""
        if not os.path.exists(filename):
            self.stdout.write(self.style.ERROR(f'File {filename} does not exist'))
            return
        
        self.stdout.write(f'Importing encryption keys from {filename}...')
        
        try:
            # Read keys from file
            with open(filename, 'r') as f:
                data = json.load(f)
            
            keys = data.get('keys', {})
            timestamp = data.get('timestamp', 'unknown')
            
            # Get all users to validate
            users = {str(user.id): user for user in User.objects.all()}
            
            # Import keys to cache
            count = 0
            for user_id, key_value in keys.items():
                # Verify user exists
                if user_id not in users:
                    self.stdout.write(self.style.WARNING(f'  User with ID {user_id} not found, skipping key'))
                    continue
                
                # Store key in cache with long timeout (30 days)
                cache_key = f"encryption_key_{user_id}"
                cache.set(cache_key, key_value, timeout=60*60*24*30)
                count += 1
                
                self.stdout.write(f'  Imported key for user {users[user_id].username} (ID: {user_id})')
            
            self.stdout.write(self.style.SUCCESS(f'Imported {count} encryption keys from {filename} (timestamp: {timestamp})'))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error importing keys: {str(e)}'))
    
    def list_keys(self):
        """List all encryption keys in cache"""
        self.stdout.write('Listing encryption keys in cache:')
        
        # Get all users
        users = {str(user.id): user for user in User.objects.all()}
        
        # Check for keys in cache
        count = 0
        for user_id, user in users.items():
            cache_key = f"encryption_key_{user_id}"
            key_value = cache.get(cache_key)
            
            if key_value:
                count += 1
                self.stdout.write(f'  User: {user.username} (ID: {user_id})')
                
                # Show key info but not the actual key
                try:
                    key_bytes = base64.b64decode(key_value.encode('utf-8'))
                    self.stdout.write(f'    - Key length: {len(key_bytes)} bytes')
                    self.stdout.write(f'    - Key prefix: {key_value[:10]}...')
                except:
                    self.stdout.write(f'    - Key (not base64): {key_value[:10]}...')
        
        if count == 0:
            self.stdout.write(self.style.WARNING('No encryption keys found in cache'))
        else:
            self.stdout.write(self.style.SUCCESS(f'Found {count} encryption keys in cache')) 