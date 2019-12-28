import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
                
#        print("****")
#        print("embed_size  :", embed_size)
#        print("hidden_size :", hidden_size)
#        print("vocab_size  :", vocab_size)
#        print("num_layers  :", num_layers)
#        print("****")
                    
        self.embed = nn.Embedding(num_embeddings=vocab_size,
                                  embedding_dim=embed_size)
#        print(self.embed)
#        print("****")        
                
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)        
#        print(self.lstm)
#        print("****")
        
        self.linear = nn.Linear(in_features=hidden_size,
                                out_features=vocab_size)
#        print(self.linear)
#        print("****")
    
    def forward(self, features, captions):
#        print("----")
#        print("features")
#        print(features.shape)    
#        print("----")
#        print("captions")
#        print(captions.shape)
        
        embeds = torch.cat((features.unsqueeze(1),self.embed(captions[:,:-1])), 1)
        results = self.linear(self.lstm(embeds)[0])        
        
        return results

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
#        print(inputs.shape)
        result = []
        for index in range(max_len): 
            out, states = self.lstm(inputs, states)
            
            out = out.squeeze(1)
            out = self.linear(out)
            res = out.max(1)[1]
            result.append(res.item())
            
            inputs = self.embed(res).unsqueeze(1)
        return result