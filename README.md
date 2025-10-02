so you can download the data through the download.py and I use Professor's token and you may need to path in download.py

<img width="1268" height="85" alt="image" src="https://github.com/user-attachments/assets/7ef2a07c-29b8-4b16-b74f-b8bcdbcb41ee" />


<img width="650" height="205" alt="image" src="https://github.com/user-attachments/assets/d3432b71-a21c-4f90-a3ab-578da72e4a7f" />
I put the file like this to avoid changing some code in train.py, so here create folder like this and download model from huggingface /uni/pytorch_model.bin.


Nearly all the files do not change. But only changing the file in hescape/experiments/configs which is the experiment config(e.g path, model, blabla) u can see my file I have edited

<img width="1666" height="397" alt="image" src="https://github.com/user-attachments/assets/d6868e4f-c7be-472a-b8d1-81663d59a073" />
So here the only thing to do is to change the csv location on your machine(for different panal, we need to change to the real csv location on our local machine,I modify the code here)
(or you can directly change the csv location in UCFhescape/experiments/configs/local_config.yaml)

(So basically you can directly change the config at hescape/experiments/configs/local_config.yaml)
<img width="854" height="280" alt="image" src="https://github.com/user-attachments/assets/7e89e6b2-fc0a-41d2-8200-28239a1d8382" />
this is just the default.
And we can modify everything at <img width="1605" height="724" alt="image" src="https://github.com/user-attachments/assets/636e1914-ebdd-4b72-836b-182740bf88ac" />

And there may be some bug when conda the enviroment, I encounter problems like two packages conflicts. See the issues I proposed in hescape(e.g drvi)
And you can apply for 4GPU

So in general. The above is just the how to change the config of the experiments

<img width="1627" height="704" alt="image" src="https://github.com/user-attachments/assets/3d8ffa94-34a2-4930-93ad-72887ab81c8e" />
for here it is in hescape/src/hescape/data_modules/image_gexp_dataset.py， we need to modify it, I have uploaded the npy file and you can check for nicheformer on github
<img width="1756" height="709" alt="image" src="https://github.com/user-attachments/assets/ecc08c98-c208-4ebf-87da-e5f375a30e46" />

yeah that may be all the problems I met when I use the hescape. If there are any more problems just message me on slack. As writing the code is my duty for the project. LoL


and the running code is  "python experiments/hescape_pretrain/train.py --config-name=local_config.yaml"



So now our goal is first reproduce the some results on benchmark and can u download lung and colon panel and use the image encoder (uni and conch) and gene encoder (nicheformer). Thanks a lot!
