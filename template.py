import torch as t
import os
from pipeline import fate_torch_hook
from fate_client.pipeline.components.fate.homo_nn import HomoNN
from fate_client.pipeline.pipeline import PipeLine
from fate_client.pipeline.components.fate.reader import Reader
from pipeline.interface import Data

fate_torch_hook(t)


fate_project_path = os.path.abspath('../../../../')
guest_0 = 10000
host_1 = 9999
pipeline = PipeLine().set_initiator(role='guest', party_id=guest_0).set_roles(guest=guest_0, host=host_1,
                                                                              arbiter=guest_0)
data_0 = {"name": "imdb", "namespace": "experiment"}
data_path = fate_project_path + '/doc/tutorial/pipeline/nn_tutorial'
pipeline.bind_table(name=data_0['name'], namespace=data_0['namespace'], path=data_path)
pipeline.bind_table(name=data_0['name'], namespace=data_0['namespace'], path=data_path)
reader_0 = Reader(name="reader_0")
reader_0.get_party_instance(role='guest', party_id=guest_0).component_param(table=data_0)
reader_0.get_party_instance(role='host', party_id=host_1).component_param(table=data_0)

reader_1 = Reader(name="reader_1")
reader_1.get_party_instance(role='guest', party_id=guest_0).component_param(table=data_0)
reader_1.get_party_instance(role='host', party_id=host_1).component_param(table=data_0)


## Add your pretriained model path here, will load model&tokenizer from this path
model_path = ''


from pipeline.component.homo_nn import DatasetParam, TrainerParam  
model = t.nn.Sequential(
    t.nn.CustModel(module_name='gpt2_multitask', class_name='MultiTaskGPT2', pretrained_path=model_path, adapter_type='HoulsbyConfig', hidden_size=768)
)

# DatasetParam
dataset_param = DatasetParam(dataset_name='multitask_ds', take_limits=50, tokenizer_name_or_path=model_path)
# TrainerParam
trainer_param = TrainerParam(trainer_name='multi_task_fedavg', epochs=1, batch_size=4, 
                             data_loader_worker=8, secure_aggregate=True)
loss = t.nn.CustLoss(loss_module_name='multi_task_loss', class_name='MultiTaskLoss', task_weights=[0.5, 0.5])


nn_component = HomoNN(name='nn_0', model=model)

# set parameter for client 1
nn_component.get_party_instance(role='guest', party_id=guest_0).component_param(
    loss=loss,
    optimizer = t.optim.Adam(lr=0.0001, eps=1e-8),
    dataset=dataset_param,       
    trainer=trainer_param,
    torch_seed=100 
)

# set parameter for client 2
nn_component.get_party_instance(role='host', party_id=host_1).component_param(
    loss=loss,
    optimizer = t.optim.Adam(lr=0.0001, eps=1e-8),
    dataset=dataset_param,       
    trainer=trainer_param,
    torch_seed=100 
)

# set parameter for server
nn_component.get_party_instance(role='arbiter', party_id=guest_0).component_param(    
    trainer=trainer_param
)

pipeline.add_component(reader_0)
pipeline.add_component(nn_component, data=Data(train_data=reader_0.output.data))
pipeline.compile()

pipeline.fit()