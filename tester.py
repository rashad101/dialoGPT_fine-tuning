from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained('output-medium')

step = 0
while True:
    userinput = input("User: ")
    if userinput=="exit":
        exit()
    input_ids = tokenizer.encode(userinput+tokenizer.eos_token, return_tensors='pt')

    bot_input_ids = torch.cat([chat_history_ids, input_ids], dim=-1) if step > 0 else input_ids

    chat_history_ids = model.generate(bot_input_ids, max_length=200, pad_token_id=tokenizer.eos_token_id, no_repeat_ngram_size=3, do_sample=True, top_k=100, top_p=0.7, temperature=0.8)
    step+=1
    print("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:,bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
