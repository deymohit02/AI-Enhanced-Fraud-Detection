"""
RabbitMQ Configuration Module
Manages RabbitMQ connections, queues, and messaging for async fraud detection
"""

import pika
import json
import os
from typing import Dict, Any, Callable
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RabbitMQConfig:
    """RabbitMQ connection and queue management"""
    
    def __init__(self):
        # Connection parameters
        self.host = os.getenv("RABBITMQ_HOST", "localhost")
        self.port = int(os.getenv("RABBITMQ_PORT", "5672"))
        self.username = os.getenv("RABBITMQ_USERNAME", "guest")
        self.password = os.getenv("RABBITMQ_PASSWORD", "guest")
        self.vhost = os.getenv("RABBITMQ_VHOST", "/")
        
        # Queue names
        self.TRANSACTION_QUEUE = "fraud_detection.transactions"
        self.ALERT_QUEUE = "fraud_detection.alerts"
        self.DLQ = "fraud_detection.dlq"  # Dead Letter Queue
        
        # Exchange
        self.EXCHANGE = "fraud_detection"
        
        # Connection
        self.connection = None
        self.channel = None
        
    def _get_connection_parameters(self):
        """Get RabbitMQ connection parameters"""
        credentials = pika.PlainCredentials(self.username, self.password)
        return pika.ConnectionParameters(
            host=self.host,
            port=self.port,
            virtual_host=self.vhost,
            credentials=credentials,
            heartbeat=600,
            blocked_connection_timeout=300
        )
    
    def connect(self):
        """Establish connection to RabbitMQ"""
        try:
            self.connection = pika.BlockingConnection(self._get_connection_parameters())
            self.channel = self.connection.channel()
            self._setup_exchanges_and_queues()
            logger.info(f"✅ Connected to RabbitMQ at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to RabbitMQ: {e}")
            logger.warning("⚠️  Running in fallback mode (no message queue)")
            return False
    
    def _setup_exchanges_and_queues(self):
        """Declare exchanges and queues"""
        
        # Declare exchange (topic exchange for routing flexibility)
        self.channel.exchange_declare(
            exchange=self.EXCHANGE,
            exchange_type='topic',
            durable=True
        )
        
        # Declare Dead Letter Queue
        self.channel.queue_declare(
            queue=self.DLQ,
            durable=True
        )
        
        # Declare Transaction Queue with DLQ
        self.channel.queue_declare(
            queue=self.TRANSACTION_QUEUE,
            durable=True,
            arguments={
                'x-dead-letter-exchange': '',
                'x-dead-letter-routing-key': self.DLQ,
                'x-message-ttl': 3600000  # 1 hour TTL
            }
        )
        
        # Bind transaction queue to exchange
        self.channel.queue_bind(
            exchange=self.EXCHANGE,
            queue=self.TRANSACTION_QUEUE,
            routing_key='transaction.#'  # Matches transaction.normal, transaction.high_priority, etc.
        )
        
        # Declare Alert Queue
        self.channel.queue_declare(
            queue=self.ALERT_QUEUE,
            durable=True
        )
        
        # Bind alert queue to exchange
        self.channel.queue_bind(
            exchange=self.EXCHANGE,
            queue=self.ALERT_QUEUE,
            routing_key='alert.#'
        )
        
        logger.info("✅ Exchanges and queues configured")
    
    def close(self):
        """Close RabbitMQ connection"""
        if self.connection and not self.connection.is_closed:
            self.connection.close()
            logger.info("🛑 RabbitMQ connection closed")


class MessageProducer:
    """RabbitMQ message producer"""
    
    def __init__(self, config: RabbitMQConfig):
        self.config = config
        if not self.config.connection or self.config.connection.is_closed:
            self.config.connect()
    
    def publish_transaction(self, transaction_data: Dict[str, Any], priority: str = "normal"):
        """
        Publish transaction to queue for async processing
        
        Args:
            transaction_data: Transaction dictionary
            priority: 'normal' or 'high_priority'
        """
        if not self.config.channel:
            logger.warning("RabbitMQ not available, skipping message publishing")
            return None
        
        routing_key = f"transaction.{priority}"
        
        try:
            # Add job ID for tracking
            import uuid
            job_id = str(uuid.uuid4())
            message = {
                "job_id": job_id,
                "data": transaction_data
            }
            
            self.config.channel.basic_publish(
                exchange=self.config.EXCHANGE,
                routing_key=routing_key,
                body=json.dumps(message),
                properties=pika.BasicProperties(
                    delivery_mode=2,  # Make message persistent
                    content_type='application/json',
                    priority=1 if priority == "high_priority" else 0
                )
            )
            
            logger.info(f"📤 Published transaction {job_id} to queue (priority: {priority})")
            return job_id
            
        except Exception as e:
            logger.error(f"Failed to publish message: {e}")
            return None
    
    def publish_alert(self, alert_data: Dict[str, Any], severity: str = "MEDIUM"):
        """
        Publish fraud alert
        
        Args:
            alert_data: Alert dictionary
            severity: LOW, MEDIUM, HIGH, CRITICAL
        """
        if not self.config.channel:
            return
        
        routing_key = f"alert.{severity.lower()}"
        
        try:
            self.config.channel.basic_publish(
                exchange=self.config.EXCHANGE,
                routing_key=routing_key,
                body=json.dumps(alert_data),
                properties=pika.BasicProperties(
                    delivery_mode=2,
                    content_type='application/json'
                )
            )
            
            logger.info(f"🚨 Published alert (severity: {severity})")
            
        except Exception as e:
            logger.error(f"Failed to publish alert: {e}")


class MessageConsumer:
    """RabbitMQ message consumer base class"""
    
    def __init__(self, config: RabbitMQConfig, queue_name: str):
        self.config = config
        self.queue_name = queue_name
        if not self.config.connection or self.config.connection.is_closed:
            self.config.connect()
    
    def start_consuming(self, callback: Callable):
        """
        Start consuming messages from queue
        
        Args:
            callback: Function to process each message
                     Should accept (ch, method, properties, body)
        """
        if not self.config.channel:
            logger.error("RabbitMQ channel not available")
            return
        
        # Set QoS (Quality of Service)
        # prefetch_count=1 ensures fair dispatch (one message at a time per worker)
        self.config.channel.basic_qos(prefetch_count=1)
        
        # Start consuming
        self.config.channel.basic_consume(
            queue=self.queue_name,
            on_message_callback=callback
        )
        
        logger.info(f"🎧 Started consuming from queue: {self.queue_name}")
        logger.info("⏳ Waiting for messages. Press CTRL+C to exit")
        
        try:
            self.config.channel.start_consuming()
        except KeyboardInterrupt:
            logger.info("⏸️  Stopping consumer...")
            self.config.channel.stop_consuming()
        except Exception as e:
            logger.error(f"Consumer error: {e}")
            self.config.channel.stop_consuming()
    
    def acknowledge_message(self, delivery_tag):
        """Acknowledge message processing"""
        if self.config.channel:
            self.config.channel.basic_ack(delivery_tag=delivery_tag)
    
    def reject_message(self, delivery_tag, requeue=False):
        """Reject message (sends to DLQ if requeue=False)"""
        if self.config.channel:
            self.config.channel.basic_nack(delivery_tag=delivery_tag, requeue=requeue)


# Singleton instance
rabbitmq_config = RabbitMQConfig()
