import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import Server
import Auxilary
from asyncio import StreamReader, StreamWriter
import asyncio
import traceback

HOST = os.environ['MODEL_HOST']
PORT = os.environ['MODEL_PORT']

MODEL = None
OPTIMIZER = None
TEMP_OUTPUT = None
TEMP_INPUT = None
TEMP_RESIDUALS = None

MAX_ERRORS = 1


def THROW_ERR(error_message : str):
    return {"ERROR": error_message}


def handle_server_request(data: dict):
    """
    Returns the data for a corresponding request from the server
    this returned data should be sent back to the server
    """
    print(f"Request Recieved: {data}")

    if 'GET_CUDA_MEM' in data:
        return {'CUDA_MEM': Auxilary.get_device_memory()}
    
    elif 'MODEL_DATA' in data:
        MODEL = data['MODEL_DATA']
        return {'MODEL_LOADED': True}
    
    elif 'OPTIMIZER' in data:
        optim = data['OPTIMIZER']
        kwargs = data['kwargs']
        kwargs['params'] = MODEL.parameters()
        OPTIMIZER = optim(kwargs)
        return None

    elif 'FORWARD' in data:
        if MODEL == None:
            return THROW_ERR("model not recieved before forward call")

        output = MODEL(data['FORWARD'])        
        return output

    elif 'BACKWARD' in data:
        OPTIMIZER.zero_grad()
        TEMP_OUTPUT.grad = data['BACKWARD']
        TEMP_OUTPUT.backward()
        OPTIMIZER.step()
        return {"BACKWARD": TEMP_INPUT.grad}

    elif 'EVAL' in data:
        MODEL.eval()
        return None

    elif 'TRAIN' in data:
        MODEL.train()
        return None

    elif 'SEND_MODEL' in data:
        return MODEL

    elif 'KILL' in data:
        return 'KILL'

    else:
        return THROW_ERR("REQUEST NOT RECOGNIZED")

async def client_loop(reader : StreamReader, writer : StreamWriter):
    ERR_COUNT = 0
    while(True):
        data = await Server.recieve_data(reader)
        try:
            response = handle_server_request(data)

            if response == 'KILL':
                print('Shutting Down worker')
                writer.close()
                await writer.wait_closed()
                break
            elif response is not None:
                await Server.send_data(writer, response)

        except:
            if (ERR_COUNT >= MAX_ERRORS):
                info = traceback.format_exc()
                await Server.send_data(writer, THROW_ERR(info))
                ERR_COUNT+=1
            else:
                await Server.send_data(writer, {"FORCE_KILL": "MAX ERRORS HIT"})
                writer.close()
                await writer.wait_closed()
                raise

async def main():
    print("Connecting to bossman")
    reader,writer = await asyncio.open_connection(HOST,PORT)
    print("connected to bossman")
    await client_loop(reader, writer)

if __name__ == "__main__":
    asyncio.run(main())

