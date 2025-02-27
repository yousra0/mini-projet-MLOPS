import os
import logging
import uuid

from datetime import datetime
from elasticsearch import Elasticsearch

from typing import Dict


logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S'
)

class LogsBuffer:
    
    DEFAULT_BUFFER_SIZE = 100
    APP_NAME = 'default-app-name'
    ENVIRONMENT_NAME = 'default-env-name'
    INDEX_NAME = 'default-index-name'

    def __init__(
        self,
        ip_addr: str,
        port: int,
        env: str,
        buffer_size: int,
        index: str,
        app_name: str
    ):
        
        self.buffer_size = int(buffer_size) if buffer_size else self.DEFAULT_BUFFER_SIZE
        self.app_name = app_name if app_name else self.APP_NAME
        self.env = env if env else self.ENVIRONMENT_NAME
        self.index = index if index else self.INDEX_NAME
        self.ip_addr = ip_addr
        self.port = port
        
        self.es_client = Elasticsearch(
            [self.ip_addr],
            port=self.port
        ) if self.ip_addr else None
        
        logging.info(f'Create log buffer [{self.env}:{self.app_name}] with size {self.buffer_size}')
        self.buffer = list()
    
    def format_log(self, msg: str, severity: str) -> Dict[str, str]:
        
        now = datetime.now()
        current_time = now.strftime('%d-%m-%Y')

        curr_index = f'{self.index}-{self.app_name}-{current_time}'

        payload = {
            'appName': self.app_name,
            'env': self.env,
            'severity': severity,
            'timestamp': now.utcnow(),
            'payload': msg,
            'id': str(uuid.uuid4())
        }

        return (curr_index, payload)
    
    @staticmethod
    def store_msgs(es_client, buffer) -> None:
        if not es_client:
            logging.warning('Elasticsearh not set up correctly use only printing to console')
        else:
            logging.debug(f'Store {len(buffer)} messages in Elasticsearh')
        
        body = list()
        for index_and_doc in buffer:
            index, doc = index_and_doc        
            body.append({'index': {'_id': doc['id']}})
            body.append(doc)
            # save messages to ES
            if es_client:
                res = es_client.bulk(index=index, body=body, doc_type='doc')
            else:
                logging.info(body)
        
    def add(self, msg: str, severity: str) -> None:
        
        logging.debug(f'Add message to buffer')
        self.buffer.append(self.format_log(msg, severity))
        logging.debug(f'Current number of messages in buffer {len(self.buffer)}')
        
        if len(self.buffer) != self.buffer_size: return
            
        LogsBuffer.store_msgs(self.es_client, self.buffer)
        self.buffer = list()
    
class ElasticLogger:
        
    def __init__(
        self,
        ip_addr: str = None,
        port: int = 9200,
        env: str = None,
        buffer_size: int = None,
        index: str = None,
        app_name: str = None
    ):
        self.logs_buffer = LogsBuffer(ip_addr, port, env, buffer_size, index, app_name)
    
    def debug(self, msg: str) -> None:
        self.logs_buffer.add(msg, severity='DEBUG')
    
    def info(self, msg: str) -> None:
        self.logs_buffer.add(msg, severity='INFO')
        
    def warning(self, msg: str) -> None:
        self.logs_buffer.add(msg, severity='WARNING')
        
    def error(self, msg: str) -> None:
        self.logs_buffer.add(msg, severity='ERROR')