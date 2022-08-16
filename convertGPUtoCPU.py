import torch
from model import generateModel

model = generateModel()
checkpoint = torch.load('./ProjectModel/model_lambda2_epoch1.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.to('cpu')


print(model)



torch.save({
    'model_state_dict': model.state_dict(),
}, './ProjectModel/model_lambda2_cpu.pt')