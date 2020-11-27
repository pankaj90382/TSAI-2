import torch
import torchaudio
import torch.nn.functional as F

classes = ['cat', 'dog', 'six', 'bird', 'eight', 'no', 'tree', 'marvin', 'left', 'down', 'off', 'on', 'five', 'three', 'go', 'seven', 'sheila', 'right', 'four', 'happy', 'bed', 'zero', 'one', 'wow', 'two', 'yes', 'house', 'up', 'nine', 'stop']
print("Number of classes", len(classes))


class SpeechRNN(torch.nn.Module):
  
  def __init__(self):
    super(SpeechRNN, self).__init__()
    
    self.lstm = torch.nn.GRU(input_size = 12, 
                              hidden_size= 256, 
                              num_layers = 2, 
                              batch_first=True)
    
    self.out_layer = torch.nn.Linear(256, 30)
    
    self.softmax = torch.nn.LogSoftmax(dim=1)
    
  def forward(self, x):
    
    out, _ = self.lstm(x)
    
    x = self.out_layer(out[:,-1,:])
    
    return self.softmax(x)


def load_audio(audio):
	with torch.no_grad():
	  # load a normalized waveform
	  waveform,_ = torchaudio.load(audio, normalization=True)
	  
	  # if the waveform is too short (less than 1 second) we pad it with zeroes
	if waveform.shape[1] < 16000:
         waveform = F.pad(input=waveform, pad=(0, 16000 - waveform.shape[1]), mode='constant', value=0)
	  
	  # then, we apply the transform
	mfcc = torchaudio.transforms.MFCC(n_mfcc=12, log_mels=True)(waveform).squeeze(0).transpose(0,1)
	return mfcc


def load_model(model):
	net = torch.load('./STT.pt')
	return net
	
def sptotex(audio, model):
	print('Loading mfcc')
	mfcc = load_audio(audio)
	print('Load Model')
	net = load_model(model)
	net.eval()
	with torch.no_grad():
		y = net(mfcc.view(1,*mfcc.shape))
	pred = y.argmax(dim=1, keepdim=True)
	return classes[pred.item()]