import sys
import socket
import struct
import pickle
import asyncio
from asyncio import StreamReader, StreamWriter

HOST = ''
PORT = 1024
OPEN_SOCKETS = []
HEADER_FORMAT = '!I'

async def send_data(writer : StreamWriter, data : object):
    pickled_data = pickle.dumps(data)
    header_data = struct.pack(HEADER_FORMAT, len(pickled_data))
    write_data = header_data + pickled_data
    writer.write(write_data)
    await writer.drain()

async def recieve_data(reader : StreamReader):
    header_size = struct.calcsize(HEADER_FORMAT)
    size_unpacked = await reader.readexactly(header_size)
    (size,) = struct.unpack(HEADER_FORMAT, size_unpacked)
    pickled_data = await reader.readexactly(size)
    return pickle.loads(pickled_data)

async def send_recieve(reader : StreamReader, writer : StreamWriter, data:object):
    await send_data(writer, data)
    response = await recieve_data(reader)
    return response

async def handle_connection(reader : StreamReader ,writer : StreamWriter):
    OPEN_SOCKETS.append((writer.get_extra_info('peername'), reader, writer))
    print(f"New Connection created , {writer.get_extra_info('peername')}")

async def start_server(server):
    try:
        await server.serve_forever()
    except asyncio.CancelledError:
        print("No longer listening for new connections.") 
        server.close()
        await server.wait_closed()
        raise

async def get_connections(num_connections : int):
    server = await asyncio.start_server(handle_connection, HOST, PORT,
                                 family=socket.AF_INET)
    print("Server Started, awaiting connections.")
    server_task = asyncio.create_task(start_server(server))

    while(True):
        await asyncio.sleep(0)
        if(len(OPEN_SOCKETS) == num_connections):
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                print("Finished Connecting with workers")
                break

async def main():
    if (len(sys.argv) == 2 and sys.argv[1] == 'server'):
        await get_connections(1)
        print(OPEN_SOCKETS)
        await send_data(OPEN_SOCKETS[0][2], "HELLO BUDDY")
    elif (len(sys.argv) == 2 and sys.argv[1] == 'client'):
        reader,writer = await asyncio.open_connection('localhost',PORT)
        data = await recieve_data(reader)
        print(data)
    else:
        print(f"USAGE: {__file__} server|client")

    
if __name__ == "__main__":
    asyncio.run(main())