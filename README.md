# CHFU (Collaborative Hierarchica Federated Unlearning)

## About The Project
**CHFU**  is provided as a robust federated unlearning technique created for IoT-FL, which consists of two main components: CH-FL and CS-AFU. 

## Getting Started

### Requirements
| Package       | Version     |
|---------------|-------------|
| torch         | 1.4.0+cu92  |
| torchvision   | 0.5.0+cu92  |
| syft          | 0.2.4       |
| python        | 3.7.16      |
| numpy         | 1.18.5      |

### File Structure
```
CHFU: Secure Federated Unlearning for IoT-FL Systems
├─data
│    └─ datapro.py  # Data downloads
│      
├─Model
│    ├─ CNNCifar.py  
│    ├─ m_MLP.py        
│    └─ m_LeNet.py         
│    
├─utils  
│    └─ Arguments.py  # Configure parameters
│    
├─ datadistri.py    
├─ unlearning.py   
├─ learning.py
├─ main.py
├─ MIA.py
└─ README.md
