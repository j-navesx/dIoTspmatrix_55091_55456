from MatrixSparseDOK import *
import machine
import network
import time
import ujson
from uuid import *
from umqtt.simple import MQTTClient
import ntptime
import uasyncio
import _thread
import ili9341
import rgb

#SETUP LCD
spi = machine.SoftSPI(baudrate=10000000, sck=machine.Pin(14), mosi=machine.Pin(13), miso=machine.Pin(12))
display = ili9341.ILI9341(spi, cs=machine.Pin(15), dc=machine.Pin(2), rst=machine.Pin(21))

# MQTT Server Parameters
id = machine.unique_id()
MQTT_CLIENT_ID = '{:02x}{:02x}{:02x}{:02x}'.format(id[0], id[1], id[2], id[3])
MQTT_BROKER = "broker.mqttdashboard.com"
MQTT_USER = ""
MQTT_PASSWORD = ""
MQTT_TOPIC_DATA = "dIoTspmatrix-data-55456-55091"
MQTT_TOPIC_CMD = "dIoTspmatrix-cmd-55456-55091"
MQTT_TOPIC_NET = "dIoTspmatrix-net-55456-55091"

# Connect to WiFi
print("Connecting to WiFi", end="")
sta_if = network.WLAN(network.STA_IF)
sta_if.active(True)
sta_if.connect('Wokwi-GUEST', '')
while not sta_if.isconnected():
  print(".", end="")
  time.sleep(0.1)
print(" Connected!")

# Connect to MQTT server
print("Connecting to MQTT server... ", end="")
client = MQTTClient(MQTT_CLIENT_ID, MQTT_BROKER, user=MQTT_USER, password=MQTT_PASSWORD)
client.connect()
print("Connected!")

#SETUP TIME
ntptime.host = "1.europe.pool.ntp.org"
try:
  ntptime.settime()
except:
  print("Error syncing time")

m1 = MatrixSparseDOK()
#m1_data = {(5, 5): 5.5, (5, 6): 5.6, (6, 7): 6.7, (7, 4): 7.4, (7, 5): 7.5, (7, 8): 7.8}
#for key, value in m1_data.items():
    #m1[Position(key[0], key[1])] = value
sum_matrix = m1
days = [m1]

year, month, mday, hour, minute, second, weekday, yearday = time.localtime()
current_day = mday
m_changed = True

def handle_interrupt(_):
    year, month, mday, hour, minute, second, weekday, yearday = time.localtime()
    days[0][Position(hour,minute)] += 1
    global m_changed
    m_changed = True


def changed_day_thread():
    global current_day
    global days
    while True:
        year, month, mday, hour, minute, second, weekday, yearday = time.localtime()
        if mday != current_day:
            days.insert(0, MatrixSparseDOK())
            current_day = mday
            print("Changed Day")
        

datapin = machine.Pin(25, machine.Pin.IN)
datapin.irq(trigger=machine.Pin.IRQ_RISING, handler=handle_interrupt)
cmd_messages = list()

msg_id = 0
node_from = 0
sum_flag = 0

def sum_all_logs():
    global msg_id
    global node_from
    year, month, mday, hour, minute, second, weekday, yearday = time.localtime()
    global sum_matrix
    m2 = ()
    try:  
        m2 = sum_matrix.compress() 
    except: 
        pass

    message = ujson.dumps({
        "cmd": "GET-LOG-FULL-EDGE",
        "data": m2,
        "log_time": f"{hour}:{minute}",
        "node_from": MQTT_CLIENT_ID,
        "node_to": node_from,
        "msg_id": msg_id,
    })
    client.publish(MQTT_TOPIC_DATA,message)

def callback(topic, message):
    global sum_flag
    topic1 = topic.decode("utf-8")
    global cmd_messages
    cmd_msg = ujson.loads(message)
    if topic1 == MQTT_TOPIC_CMD:
        if not cmd_msg["node_from"] == MQTT_CLIENT_ID:
            if cmd_msg["node_to"] == MQTT_CLIENT_ID or cmd_msg["node_to"] == "ANY":
                cmd_messages.append(cmd_msg)
            process_commands()
    if topic1 == MQTT_TOPIC_DATA and cmd_msg["node_to"] == MQTT_CLIENT_ID and sum_flag == 1:
        global sum_matrix
        data = (tuple(cmd_msg["data"][0]), cmd_msg["data"][1], tuple(cmd_msg["data"][2]), tuple(cmd_msg["data"][3]), tuple(cmd_msg["data"][4])) 
        sum_matrix += MatrixSparseDOK.decompress(data)
        
    
client.set_callback(callback)
client.subscribe(MQTT_TOPIC_CMD)
client.subscribe(MQTT_TOPIC_DATA)

def process_commands():
    global cmd_messages
    for i in range(len(cmd_messages)):
        year, month, mday, hour, minute, second, weekday, yearday = time.localtime()
        cmd_msg = cmd_messages[0]
        m2 = ()
        if cmd_msg["cmd"] == "GET-NODE-LOG-FULL":
            try:  
                m2 = days[-cmd_msg["day"]].compress()
            except: 
                m2 = ()             
            message = ujson.dumps({
                "cmd": "GET-NODE-LOG-FULL",
                "node_from": MQTT_CLIENT_ID,
                "node_to": cmd_msg["node_from"],
                "data": m2,
                "msg_id": cmd_msg["msg_id"],
                "log_time": f"{hour}:{minute}"
            })
            client.publish(MQTT_TOPIC_DATA, message)
        if cmd_msg["cmd"] == "GET-NODE-LOG-BY-HOUR":

            try:  
                m2 = days[-cmd_msg["day"]].row(hour)
                m2 = m2.compress()      
            except: 
                m2 = ()              
            hour = cmd_msg["hour"]
            message = ujson.dumps({
                "cmd": "GET-NODE-LOG-BY-HOUR",
                "data": m2,
                "log_time": f"{hour}:{minute}",
                "node_from": MQTT_CLIENT_ID,
                "node_to": cmd_msg["node_from"],
                "msg_id": cmd_msg["msg_id"],
            })
            client.publish(MQTT_TOPIC_DATA, message)
        if cmd_msg["cmd"] == "GET-NODE-LOG-BY-MINUTE":

            try:  
                m2 = days[-cmd_msg["day"]].col(minute)
                m2 = m2.compress()       
            except: 
                m2 = ()  

            minute = cmd_msg["minute"]
            message = ujson.dumps({
                "cmd": "GET-NODE-LOG-BY-MINUTE",
                "data": m2,
                "log_time": f"{hour}:{minute}",
                "node_from": MQTT_CLIENT_ID,
                "node_to": cmd_msg["node_from"],
                "msg_id": cmd_msg["msg_id"],
            })
            client.publish(MQTT_TOPIC_DATA, message)
        if cmd_msg["cmd"] == "GET-LOG-FULL-EDGE":
            global sum_matrix
            global sum_flag
            sum_flag = 1
            sum_matrix = days[-cmd_msg["day"]]
            message = ujson.dumps({
                "cmd": "GET-NODE-LOG-FULL",
                "day": cmd_msg["day"],
                "node_from": MQTT_CLIENT_ID,
                "node_to": "ANY",
                "msg_id": cmd_msg["msg_id"]
            })
            global msg_id 
            global node_from
            msg_id = cmd_msg["msg_id"]
            node_from = cmd_msg["node_from"]
            client.publish(MQTT_TOPIC_CMD,message)
        cmd_messages.pop(0)

last_minute = -1

def alive_thread():
    while True:
        global sum_flag
        year, month, mday, hour, minute, second, weekday, yearday = time.localtime()
        global last_minute
        if(minute != last_minute):
            year, month, mday, hour, minute, second, weekday, yearday = time.localtime()
            message = ujson.dumps({
                "cmd": "NODE ALIVE",
                "node_id_from": MQTT_CLIENT_ID,
                "timestamp": f"{year}-{month}-{mday} {hour}:{minute}:{second}"
            })
            client.publish(MQTT_TOPIC_NET, message)
            last_minute = minute
            if sum_flag:
                sum_all_logs()
            sum_flag = 0

def mqtt_thread():
    while True:
        client.check_msg()
        process_commands()

def display_thread():
    width=240
    height=320
    while True: 
        global m_changed
        if m_changed and len(days[0]):
            length = len(days[0])
            m_changed = False
            (min_row, min_col), (max_row, max_col) = days[0].dim()
            sep_color = int(255/length)
            sep_width = int(width/max_col)
            sep_height = int(height/max_row)
            i = 0
            for key,value in sorted(days[0]._items.items()):
                x = sep_width*(key[1]-min_col)
                y = sep_height*(key[0]-min_row)
                display.fill_rectangle(x,y,10,10,rgb.color565(255-(i*sep_color), 0, 0))
                i += 1

_thread.start_new_thread(alive_thread,())
_thread.start_new_thread(mqtt_thread,())
_thread.start_new_thread(display_thread,())
_thread.start_new_thread(changed_day_thread,())
