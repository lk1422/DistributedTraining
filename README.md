# DistributedTraining
Implementing a method of training across multiple remote devices (with GPUs). The Naive_Parallel will not feature any pipelining, after the program is able to support basic model parallel training, I will move onto developing support for data parallel training.


Simple Model Implementation:
  -Residual Connections and complicated inputs can be handled by wrapping a class and adding variables for in_variables and out_variables.
  -in_variables are used for extra incoming inputs (masks, residual connections etc)
  -out_variables are used for outgoing variables which will be used by a future module.
  -The system supports input/output connections between modules which are not stored on the same device
Here is an example residual

    TEST_RESIDUAL_SMALL = [
          (nn.Linear, {'in_features':1, 'out_features':512}, (32,  1)),
          (nn.ReLU,   {}, (32,512)),
          (wrapped_linear, {
              'in_features':512,
              'out_features':512,
              'out_variables': ["res_1"]
              }, [(32, 512)]),
          (nn.ReLU,   {}, (32,512)),
          (nn.Linear, {'in_features':512, 'out_features':512}, (32,512)),
          (nn.ReLU,   {}, (32,512)),
          (wrapped_linear, {
              'in_features':512,
              'out_features':512,
              'dependencies':["res_1"],
              'out_variables': []
          }, [(32, 512), (32,512)]),
          (nn.ReLU,   {}, (32,512)),
          (nn.Linear, {'in_features':512, 'out_features':1}, (32, 512)),
          (nn.Sigmoid,   {}, (32,1))
      ]

  Simple Train Loop:
    -The training loop has a very close resemblance to the pytorch boilerplate train loop:

    #Set up server and await connections
    workers = await BossMan.setup_host(num_workers)
    boss = BossMan(workers, model)
    await boss.distribute_model()
    await boss.send_optimizer(torch.optim.Adam, {"lr":3e-3})
    boss.gather_dependencies()
    #Server set up begin regular training
    loss_fn = torch.nn.MSELoss()
    EPOCHS = 400
    for _ in range(EPOCHS):
        x,y = get_batch(32)
        prediction = await boss.forward(x, {})
        loss = loss_fn(prediction, y)
        loss.backward()
        await boss.backward(prediction.grad)
        await boss.optim_step()

  Usage:
    After you have created a model which the bossman can process and initialized the bossman such as the examples above
    just run worker.py on the worker devices and the program will execute as usual. Just remember for first set the the following enviorment variables.
    MODEL_HOST="123.456.789" (IP of the device running  the bossman code)
    MODEL_PORT=1024 (Constant do not change)
    
    

   
