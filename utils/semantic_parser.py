#!/usr/bin/env python3
"""
Semantic Parser for Provenance Data

This module provides a universal semantic parser that can automatically identify
and parse different provenance data formats (DARPA CDM, Elastic, custom JSON, etc.)
and convert them into a unified internal representation.
"""

import json
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
import struct

# Try to import Avro libraries for binary file support
try:
    import avro.datafile
    import avro.io
    AVRO_AVAILABLE = True
except ImportError:
    AVRO_AVAILABLE = False
    logging.warning("avro-python3 not available, binary AVRO files will not be supported")

try:
    import fastavro
    FASTAVRO_AVAILABLE = True
except ImportError:
    FASTAVRO_AVAILABLE = False

logger = logging.getLogger(__name__)


class Entity:
    """Unified entity representation"""
    
    def __init__(self, entity_id: str, entity_type: str, attributes: Dict = None):
        self.entity_id = entity_id
        self.entity_type = entity_type
        self.attributes = attributes or {}
        self.creation_time = None
        
    def __repr__(self):
        return f"Entity({self.entity_type}:{self.entity_id})"


class Event:
    """Unified event representation"""
    
    def __init__(self, event_id: str, event_type: str, 
                 source: Entity, target: Entity,
                 timestamp: float, attributes: Dict = None):
        self.event_id = event_id
        self.event_type = event_type
        self.source = source
        self.target = target
        self.timestamp = timestamp
        self.attributes = attributes or {}
        
    def __repr__(self):
        return f"Event({self.event_type}: {self.source} -> {self.target})"


class BaseParser(ABC):
    """Abstract base parser for provenance data"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.entity_registry = {}  # Map entity IDs to Entity objects
        self.events = []
        self.statistics = defaultdict(int)
        
    @abstractmethod
    def can_parse(self, data_sample: Any) -> bool:
        """Check if this parser can handle the given data format"""
        pass
    
    @abstractmethod
    def parse_event(self, raw_event: Any) -> Optional[Event]:
        """Parse a single event from raw format"""
        pass
    
    def get_or_create_entity(self, entity_id: str, entity_type: str, 
                            attributes: Dict = None) -> Entity:
        """Get existing entity or create new one"""
        key = f"{entity_type}:{entity_id}"
        
        if key not in self.entity_registry:
            entity = Entity(entity_id, entity_type, attributes)
            self.entity_registry[key] = entity
            self.statistics[f'entity_{entity_type}'] += 1
        else:
            entity = self.entity_registry[key]
            # Update attributes if new ones provided
            if attributes:
                entity.attributes.update(attributes)
                
        return entity
    
    def parse_timestamp(self, timestamp: Any) -> float:
        """Parse various timestamp formats to Unix timestamp"""
        if isinstance(timestamp, (int, float)):
            # Assume already Unix timestamp, but check if it's in nanoseconds
            if timestamp > 1e15:  # Likely nanoseconds
                return timestamp / 1e9
            elif timestamp > 1e12:  # Likely milliseconds
                return timestamp / 1e3
            return float(timestamp)
        
        if isinstance(timestamp, str):
            try:
                # Try ISO format
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                return dt.timestamp()
            except:
                try:
                    # Try parsing as float
                    return float(timestamp)
                except:
                    logger.warning(f"Could not parse timestamp: {timestamp}")
                    return 0.0
        
        return 0.0
    
    def get_statistics(self) -> Dict:
        """Get parsing statistics"""
        return dict(self.statistics)


class DARPACDMParser(BaseParser):
    """Parser for DARPA TC CDM format (AVRO)"""
    
    CDM_VERSION = "18"  # Support CDM version 18
    
    # CDM entity type mappings
    ENTITY_TYPE_MAP = {
        "SUBJECT": "process",
        "FILE_OBJECT": "file",
        "NETFLOW_OBJECT": "network",
        "MEMORY_OBJECT": "memory",
        "SRC_SINK_OBJECT": "socket",
        "PRINCIPAL": "principal",
        "REGISTRY_KEY_OBJECT": "registry"
    }
    
    # CDM event type mappings
    EVENT_TYPE_MAP = {
        "EVENT_EXECUTE": "exec",
        "EVENT_FORK": "fork",
        "EVENT_CLONE": "clone",
        "EVENT_EXIT": "exit",
        "EVENT_READ": "read",
        "EVENT_WRITE": "write",
        "EVENT_OPEN": "open",
        "EVENT_CLOSE": "close",
        "EVENT_MMAP": "mmap",
        "EVENT_CONNECT": "connect",
        "EVENT_ACCEPT": "accept",
        "EVENT_SENDTO": "sendto",
        "EVENT_RECVFROM": "recvfrom",
        "EVENT_SENDMSG": "sendmsg",
        "EVENT_RECVMSG": "recvmsg",
        "EVENT_RENAME": "rename",
        "EVENT_LINK": "link",
        "EVENT_UNLINK": "unlink",
        "EVENT_CREATE_OBJECT": "create",
        "EVENT_MODIFY_FILE_ATTRIBUTES": "modify_attr",
        "EVENT_LOADLIBRARY": "loadlib",
        "EVENT_OTHER": "other"
    }
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.uuid_to_entity = {}  # Map CDM UUIDs to Entity objects
        
    def can_parse(self, data_sample: Any) -> bool:
        """Check if data is in DARPA CDM format"""
        if not isinstance(data_sample, dict):
            return False
        
        # Check for CDM structure
        has_datum = 'datum' in data_sample
        has_cdm_version = 'CDMVersion' in data_sample
        has_source = 'source' in data_sample
        
        # Check if datum contains CDM schema
        if has_datum:
            datum = data_sample.get('datum', {})
            cdm_schemas = [
                'com.bbn.tc.schema.avro.cdm18.Event',
                'com.bbn.tc.schema.avro.cdm18.Subject',
                'com.bbn.tc.schema.avro.cdm18.FileObject',
                'com.bbn.tc.schema.avro.cdm18.NetFlowObject',
                'com.bbn.tc.schema.avro.cdm18.Host',
                'com.bbn.tc.schema.avro.cdm18.Principal'
            ]
            for schema in cdm_schemas:
                if schema in datum:
                    return True
        
        return has_datum and (has_cdm_version or has_source)
    
    def parse_event(self, raw_event: Dict) -> Optional[Event]:
        """Parse DARPA CDM event"""
        try:
            datum = raw_event.get('datum', {})
            
            # Extract the actual record from the datum wrapper
            record = None
            record_type = None
            
            # Check for nested schema format (JSON NDJSON files)
            for key, value in datum.items():
                if key.startswith('com.bbn.tc.schema.avro.cdm'):
                    record = value
                    record_type = key.split('.')[-1]
                    break
            
            # If no nested schema, check if datum itself is the record (binary AVRO format)
            if not record and datum:
                # Detect record type from fields present
                if 'type' in datum and datum.get('type') in self.EVENT_TYPE_MAP:
                    record = datum
                    record_type = 'Event'
                elif 'pid' in datum or 'ppid' in datum or 'cid' in datum or 'cmdLine' in datum or 'parentSubject' in datum:
                    # Subject (process) - can have pid/ppid (older format) or cid/cmdLine (CDM 18)
                    record = datum
                    record_type = 'Subject'
                elif 'fileDescriptor' in datum or ('baseObject' in datum and 'path' in str(datum.get('baseObject', {}))):
                    record = datum
                    record_type = 'FileObject'
                elif 'srcAddress' in datum and 'destAddress' in datum:
                    record = datum
                    record_type = 'NetFlowObject'
                elif 'hostName' in datum and 'hostType' in datum:
                    record = datum
                    record_type = 'Host'
                elif 'username' in datum:
                    record = datum
                    record_type = 'Principal'
                else:
                    # Unknown record type
                    self.statistics['unknown_records'] += 1
                    return None
            
            if not record:
                return None
            
            # Track record types for debugging
            self.statistics[f'record_type_{record_type}'] += 1
            
            # Process different CDM record types
            if record_type == 'Event':
                return self._parse_cdm_event(record)
            elif record_type in ['Subject', 'FileObject', 'NetFlowObject', 
                                'MemoryObject', 'SrcSinkObject', 'Principal']:
                # These are entity definitions, register them
                self._register_cdm_entity(record, record_type)
                return None
            elif record_type == 'Host':
                # Host information, can be used for context
                self._register_host_info(record)
                return None
            
        except Exception as e:
            logger.debug(f"Error parsing CDM record: {e}")
            self.statistics['parse_errors'] += 1
        
        return None
    
    def _parse_cdm_event(self, record: Dict) -> Optional[Event]:
        """Parse CDM Event record"""
        # Extract basic event information
        event_uuid = record.get('uuid', '')
        event_type_cdm = record.get('type', 'EVENT_OTHER')
        event_type = self.EVENT_TYPE_MAP.get(event_type_cdm, 'other')
        
        # Get timestamp (in nanoseconds typically)
        timestamp_ns = record.get('timestampNanos', 0)
        timestamp = self.parse_timestamp(timestamp_ns)
        
        # Extract subject (source entity - typically a process)
        subject_uuid = record.get('subject', {})
        if isinstance(subject_uuid, dict):
            subject_uuid = subject_uuid.get('com.bbn.tc.schema.avro.cdm18.UUID', '')
        
        # Extract predicateObject (target entity)
        predicate_object = record.get('predicateObject', {})
        if isinstance(predicate_object, dict):
            predicate_object = predicate_object.get('com.bbn.tc.schema.avro.cdm18.UUID', '')
        
        # Extract predicateObject2 (for events with two objects)
        predicate_object2 = record.get('predicateObject2')
        if predicate_object2 and isinstance(predicate_object2, dict):
            predicate_object2 = predicate_object2.get('com.bbn.tc.schema.avro.cdm18.UUID', '')
        
        # Get or create entities
        source_entity = self._get_entity_by_uuid(subject_uuid)
        target_entity = self._get_entity_by_uuid(predicate_object)
        
        if not source_entity or not target_entity:
            self.statistics['events_missing_entities'] += 1
            return None
        
        # Create event
        event = Event(
            event_id=event_uuid,
            event_type=event_type,
            source=source_entity,
            target=target_entity,
            timestamp=timestamp,
            attributes={
                'cdm_type': event_type_cdm,
                'predicate_object2': predicate_object2,
                'thread_id': record.get('threadId'),
                'sequence': record.get('sequence'),
                'properties': record.get('properties', {})
            }
        )
        
        self.statistics['events_parsed'] += 1
        self.statistics[f'event_{event_type}'] += 1
        
        return event
    
    def _register_cdm_entity(self, record: Dict, record_type: str):
        """Register CDM entity in the registry"""
        uuid = record.get('uuid', '')
        if not uuid:
            return
        
        # Determine entity type
        entity_type = self.ENTITY_TYPE_MAP.get(
            record.get('type', record_type.upper()), 
            record_type.lower()
        )
        
        # Extract entity attributes based on type
        attributes = {}
        
        if record_type == 'Subject':  # Process
            # Handle both old format (pid/ppid) and CDM 18 format (cid)
            pid = record.get('pid') or record.get('cid')
            ppid = record.get('ppid')
            
            attributes = {
                'pid': pid,
                'ppid': ppid,
                'cid': record.get('cid'),
                'cmdline': record.get('cmdLine'),
                'properties': record.get('properties', {}),
                'local_principal': record.get('localPrincipal'),
                'parent_subject': record.get('parentSubject')
            }
            
            # Create entity ID from pid/cid
            if pid:
                entity_id = f"pid_{pid}_{str(uuid)[:8] if isinstance(uuid, str) else uuid}"
            else:
                entity_id = str(uuid)
            
        elif record_type == 'FileObject':
            attributes = {
                'path': record.get('baseObject', {}).get('properties', {}).get('path', ''),
                'file_descriptor': record.get('fileDescriptor'),
                'properties': record.get('baseObject', {}).get('properties', {})
            }
            entity_id = attributes.get('path', uuid)
            
        elif record_type == 'NetFlowObject':
            attributes = {
                'src_address': record.get('srcAddress'),
                'src_port': record.get('srcPort'),
                'dest_address': record.get('destAddress'),
                'dest_port': record.get('destPort'),
                'protocol': record.get('ipProtocol')
            }
            entity_id = f"{attributes['src_address']}:{attributes['src_port']}-{attributes['dest_address']}:{attributes['dest_port']}"
            
        else:
            entity_id = uuid
            attributes = record
        
        # Create and register entity
        entity = self.get_or_create_entity(entity_id, entity_type, attributes)
        self.uuid_to_entity[uuid] = entity
    
    def _get_entity_by_uuid(self, uuid: str) -> Optional[Entity]:
        """Get entity by CDM UUID"""
        return self.uuid_to_entity.get(uuid)
    
    def _register_host_info(self, record: Dict):
        """Register host information"""
        self.statistics['host_info'] = {
            'hostname': record.get('hostName', ''),
            'uuid': record.get('uuid', ''),
            'os': record.get('osDetails', ''),
            'interfaces': record.get('interfaces', [])
        }


class ElasticParser(BaseParser):
    """Parser for Elastic/ELK stack logs"""
    
    def can_parse(self, data_sample: Any) -> bool:
        """Check if data is in Elastic format"""
        if not isinstance(data_sample, dict):
            return False
        
        # Check for Elastic-specific fields
        has_timestamp = '@timestamp' in data_sample
        has_event = 'event' in data_sample
        has_data_stream = 'data_stream' in data_sample
        has_ecs_version = 'ecs' in data_sample
        
        return has_timestamp and (has_event or has_data_stream or has_ecs_version)
    
    def parse_event(self, raw_event: Dict) -> Optional[Event]:
        """Parse Elastic event"""
        try:
            # Determine event category
            event_info = raw_event.get('event', {})
            event_action = event_info.get('action', 'unknown')
            event_category = event_info.get('category', ['unknown'])
            if isinstance(event_category, list):
                event_category = event_category[0] if event_category else 'unknown'
            
            # Extract timestamp
            timestamp_str = raw_event.get('@timestamp', '')
            timestamp = self.parse_timestamp(timestamp_str)
            
            # Extract process information (source)
            process = raw_event.get('process', {})
            source_id = process.get('entity_id') or f"{process.get('executable', 'unknown')}:{process.get('pid', '')}"
            source_entity = self.get_or_create_entity(
                source_id,
                'process',
                {
                    'name': process.get('name', ''),
                    'executable': process.get('executable', ''),
                    'pid': process.get('pid', ''),
                    'ppid': process.get('parent', {}).get('pid', '')
                }
            )
            
            # Extract target based on event category
            target_entity = None
            
            if event_category == 'file':
                file_info = raw_event.get('file', {})
                file_path = file_info.get('path', file_info.get('name', ''))
                target_entity = self.get_or_create_entity(
                    file_path,
                    'file',
                    {
                        'path': file_path,
                        'name': file_info.get('name', ''),
                        'extension': file_info.get('extension', '')
                    }
                )
                
            elif event_category == 'network':
                destination = raw_event.get('destination', {})
                dest_ip = destination.get('ip', '')
                dest_port = destination.get('port', '')
                network_id = f"{dest_ip}:{dest_port}" if dest_port else dest_ip
                
                target_entity = self.get_or_create_entity(
                    network_id,
                    'network',
                    {
                        'dest_ip': dest_ip,
                        'dest_port': dest_port,
                        'src_ip': raw_event.get('source', {}).get('ip', ''),
                        'src_port': raw_event.get('source', {}).get('port', ''),
                        'transport': raw_event.get('network', {}).get('transport', '')
                    }
                )
                
            elif event_category == 'process':
                parent = process.get('parent', {})
                parent_id = parent.get('entity_id') or f"{parent.get('executable', 'unknown')}:{parent.get('pid', '')}"
                target_entity = self.get_or_create_entity(
                    parent_id,
                    'process',
                    {
                        'name': parent.get('name', ''),
                        'executable': parent.get('executable', ''),
                        'pid': parent.get('pid', '')
                    }
                )
            
            if not target_entity:
                self.statistics['events_missing_target'] += 1
                return None
            
            # Create event
            event = Event(
                event_id=f"{timestamp}_{source_id}_{event_action}",
                event_type=event_action,
                source=source_entity,
                target=target_entity,
                timestamp=timestamp,
                attributes={
                    'category': event_category,
                    'user': raw_event.get('user', {}).get('name', ''),
                    'host': raw_event.get('host', {}).get('hostname', ''),
                    'outcome': event_info.get('outcome', '')
                }
            )
            
            self.statistics['events_parsed'] += 1
            self.statistics[f'event_{event_category}'] += 1
            
            return event
            
        except Exception as e:
            logger.debug(f"Error parsing Elastic event: {e}")
            self.statistics['parse_errors'] += 1
            return None


class CustomJSONParser(BaseParser):
    """Generic parser for custom JSON formats"""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.schema_config = config.get('schema', {}) if config else {}
        
    def can_parse(self, data_sample: Any) -> bool:
        """Check if data is valid JSON with required fields"""
        if not isinstance(data_sample, dict):
            return False
        
        # If schema is provided in config, validate against it
        if self.schema_config:
            required_fields = self.schema_config.get('required_fields', [])
            return all(field in data_sample for field in required_fields)
        
        # Generic check - must have some basic structure
        return len(data_sample) > 0
    
    def parse_event(self, raw_event: Dict) -> Optional[Event]:
        """Parse custom JSON event"""
        try:
            # Use schema config to extract fields
            timestamp_field = self.schema_config.get('timestamp_field', 'timestamp')
            source_field = self.schema_config.get('source_field', 'source')
            target_field = self.schema_config.get('target_field', 'target')
            event_type_field = self.schema_config.get('event_type_field', 'type')
            
            timestamp = self.parse_timestamp(raw_event.get(timestamp_field, 0))
            
            # Extract source
            source_data = raw_event.get(source_field, {})
            if isinstance(source_data, str):
                source_id = source_data
                source_type = 'entity'
            else:
                source_id = source_data.get('id', source_data.get('name', 'unknown'))
                source_type = source_data.get('type', 'entity')
            
            source_entity = self.get_or_create_entity(source_id, source_type, source_data if isinstance(source_data, dict) else {})
            
            # Extract target
            target_data = raw_event.get(target_field, {})
            if isinstance(target_data, str):
                target_id = target_data
                target_type = 'entity'
            else:
                target_id = target_data.get('id', target_data.get('name', 'unknown'))
                target_type = target_data.get('type', 'entity')
            
            target_entity = self.get_or_create_entity(target_id, target_type, target_data if isinstance(target_data, dict) else {})
            
            # Event type
            event_type = raw_event.get(event_type_field, 'unknown')
            
            # Create event
            event = Event(
                event_id=f"{timestamp}_{source_id}_{event_type}",
                event_type=event_type,
                source=source_entity,
                target=target_entity,
                timestamp=timestamp,
                attributes=raw_event
            )
            
            self.statistics['events_parsed'] += 1
            return event
            
        except Exception as e:
            logger.debug(f"Error parsing custom JSON event: {e}")
            self.statistics['parse_errors'] += 1
            return None


class SemanticParser:
    """
    Universal semantic parser that automatically detects and parses
    different provenance data formats.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Initialize available parsers
        self.parsers = [
            DARPACDMParser(config),
            ElasticParser(config),
            CustomJSONParser(config)
        ]
        
        self.selected_parser = None
        self.events = []
        
    def auto_detect_format(self, data_sample: Any) -> Optional[BaseParser]:
        """Automatically detect data format and select appropriate parser"""
        for parser in self.parsers:
            if parser.can_parse(data_sample):
                logger.info(f"Detected format: {parser.__class__.__name__}")
                return parser
        
        logger.warning("Could not auto-detect data format, using CustomJSONParser as fallback")
        return self.parsers[-1]  # Return CustomJSONParser as fallback
    
    def parse_file(self, file_path: Path, max_events: int = None) -> List[Event]:
        """
        Parse a provenance data file (JSON, NDJSON, or binary AVRO).
        
        Args:
            file_path: Path to file (JSON, NDJSON, or binary AVRO)
            max_events: Maximum number of events to parse (None = all)
            
        Returns:
            List of parsed Event objects
        """
        logger.info(f"Parsing file: {file_path}")
        
        # Check file extension and name to determine format
        file_ext = file_path.suffix.lower()
        file_name = file_path.name.lower()
        # Check if file name contains .bin (handles .bin, .bin.1, .bin.2, etc.)
        is_binary = file_ext in ['.bin', '.avro', '.dat'] or '.bin' in file_name
        
        # Try binary AVRO first if it looks like binary
        if is_binary:
            if FASTAVRO_AVAILABLE or AVRO_AVAILABLE:
                try:
                    return self._parse_avro_file(file_path, max_events)
                except Exception as e:
                    logger.warning(f"Failed to parse as AVRO: {e}, trying JSON fallback")
            else:
                logger.error(f"Binary AVRO file detected but no AVRO library available!")
                logger.error(f"Please install: pip install fastavro avro-python3")
                return []
        
        # Parse as JSON/NDJSON
        return self._parse_json_file(file_path, max_events)
    
    def _parse_json_file(self, file_path: Path, max_events: int = None) -> List[Event]:
        """Parse JSON or NDJSON file"""
        events = []
        lines_processed = 0
        
        try:
            with open(file_path, 'r') as f:
                # Read first line to detect format
                first_line = f.readline().strip()
                if not first_line:
                    logger.warning(f"Empty file: {file_path}")
                    return events
                
                # Try to parse as JSON
                try:
                    first_record = json.loads(first_line)
                    
                    # Auto-detect format
                    if not self.selected_parser:
                        self.selected_parser = self.auto_detect_format(first_record)
                    
                    # Parse first record
                    event = self.selected_parser.parse_event(first_record)
                    if event:
                        events.append(event)
                    
                    lines_processed += 1
                    
                    # Continue parsing rest of file
                    for line in f:
                        if max_events and lines_processed >= max_events:
                            break
                        
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            record = json.loads(line)
                            event = self.selected_parser.parse_event(record)
                            if event:
                                events.append(event)
                            
                            lines_processed += 1
                            
                            if lines_processed % 10000 == 0:
                                logger.info(f"Processed {lines_processed} lines, extracted {len(events)} events")
                                
                        except json.JSONDecodeError:
                            logger.debug(f"Skipping invalid JSON line")
                            continue
                    
                except json.JSONDecodeError:
                    logger.error(f"File is not in JSON format: {file_path}")
                    return events
        
        except Exception as e:
            logger.error(f"Error parsing JSON file {file_path}: {e}")
            return events
        
        logger.info(f"✓ Parsed {len(events)} events from {lines_processed} records")
        
        return events
    
    def _parse_avro_file(self, file_path: Path, max_events: int = None) -> List[Event]:
        """Parse binary AVRO file (DARPA TC format)"""
        logger.info(f"Parsing AVRO file: {file_path}")
        
        events = []
        records_processed = 0
        
        try:
            # Try fastavro first (faster)
            if FASTAVRO_AVAILABLE:
                with open(file_path, 'rb') as f:
                    reader = fastavro.reader(f)
                    
                    for record in reader:
                        if max_events and records_processed >= max_events:
                            break
                        
                        # Auto-detect format if not already done
                        if not self.selected_parser and records_processed == 0:
                            self.selected_parser = self.auto_detect_format(record)
                        
                        # Parse record
                        event = self.selected_parser.parse_event(record)
                        if event:
                            events.append(event)
                        
                        records_processed += 1
                        
                        if records_processed % 10000 == 0:
                            logger.info(f"Processed {records_processed} records, extracted {len(events)} events")
            
            # Fallback to avro-python3
            elif AVRO_AVAILABLE:
                with open(file_path, 'rb') as f:
                    reader = avro.datafile.DataFileReader(f, avro.io.DatumReader())
                    
                    for record in reader:
                        if max_events and records_processed >= max_events:
                            break
                        
                        # Auto-detect format if not already done
                        if not self.selected_parser and records_processed == 0:
                            self.selected_parser = self.auto_detect_format(record)
                        
                        # Parse record
                        event = self.selected_parser.parse_event(record)
                        if event:
                            events.append(event)
                        
                        records_processed += 1
                        
                        if records_processed % 10000 == 0:
                            logger.info(f"Processed {records_processed} records, extracted {len(events)} events")
                    
                    reader.close()
        
        except Exception as e:
            logger.error(f"Error parsing AVRO file {file_path}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return events
        
        logger.info(f"✓ Parsed {len(events)} events from {records_processed} AVRO records")
        
        return events
    
    def get_statistics(self) -> Dict:
        """Get parsing statistics"""
        if self.selected_parser:
            return self.selected_parser.get_statistics()
        return {}
    
    def get_entity_registry(self) -> Dict:
        """Get all registered entities"""
        if self.selected_parser:
            return self.selected_parser.entity_registry
        return {}


def parse_provenance_data(file_path: Path, config: Dict = None, 
                         max_events: int = None) -> Tuple[List[Event], Dict]:
    """
    Convenience function to parse provenance data from a file.
    
    Args:
        file_path: Path to provenance data file
        config: Optional parser configuration
        max_events: Maximum number of events to parse
        
    Returns:
        Tuple of (events list, statistics dict)
    """
    parser = SemanticParser(config)
    events = parser.parse_file(file_path, max_events)
    statistics = parser.get_statistics()
    
    return events, statistics


if __name__ == '__main__':
    # Test the parser
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 1:
        test_file = Path(sys.argv[1])
        events, stats = parse_provenance_data(test_file, max_events=1000)
        
        print(f"\n{'='*80}")
        print(f"Parsed {len(events)} events")
        print(f"{'='*80}")
        print("\nStatistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    else:
        print("Usage: python semantic_parser.py <provenance_file>")
