from pika import BlockingConnection, PlainCredentials, ConnectionParameters
from json import loads
from sys import exit

connection: BlockingConnection

def queue_callback(channel, method, properties, body):
    decoded_body = body.decode('UTF-8')
    json_body = loads(decoded_body)
    
    file_contents_raw = json_body['data']['command']
    print(f"message {file_contents_raw} received from channel {channel} with method {method}")

def signal_handler(signal, frame):
  print("\nCTRL-C handler, cleaning up rabbitmq connection and quitting")
  connection.close()
  exit(0)

def register_queue():
  # connect to RabbitMQ
  credentials = PlainCredentials('guest', 'guest')
  connection = BlockingConnection(ConnectionParameters(host = 'rabbitmq', port=5672, credentials = credentials))
  
  channel = connection.channel()
  channel.basic_consume(queue='job', on_message_callback=queue_callback, auto_ack=True)
  channel.start_consuming()

if __name__ == "__main__":
    # execute only if run as a script
    register_queue()