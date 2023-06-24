import torch
import torch.nn as nn
from bossman import BossMan
import Server
import asyncio
import matplotlib.pyplot as plt
from typing import Tuple, List
from test import *

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def get_batch(batch_size):
    x = torch.rand(batch_size,1)
    y = torch.sin(x)
    return(x,y)

async def set_up_boss(num_workers, model):
    workers = await BossMan.setup_host(num_workers)
    boss = BossMan(workers, model)
    await boss.distribute_model()
    await boss.send_optimizer(torch.optim.Adam, {"lr":3e-3})
    return boss
    

async def test_train():
    EPOCHS = 400
    boss = await set_up_boss(2, TEST_RESIDUAL_SMALL)
    boss.gather_dependencies()
    loss_fn = torch.nn.MSELoss()
    losses = []
    for e in range(EPOCHS):
        x,y = get_batch(32)
        prediction = await boss.forward(x, {})
        loss = loss_fn(prediction, y)
        loss.backward()
        await boss.backward(prediction.grad)
        await boss.optim_step()
        losses.append(loss.item())

    x = [i for i in range(EPOCHS)]
    plt.plot(x,losses)
    plt.show()

    CLOSE_CONNECTION = {'KILL': True}
    for worker in boss.workers:
        await Server.send_data(worker.writer, CLOSE_CONNECTION)
        worker.writer.close()
        await worker.writer.wait_closed()

if __name__ == "__main__":
    asyncio.run(test_train())

