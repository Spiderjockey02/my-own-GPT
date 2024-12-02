from helpers.GPT import GPTLanguageModel
from helpers.Tokenizer import TokenProcessor


while True:
    text = input("Command: ")
    if text == "token":
        processor = TokenProcessor('./data/text.txt')
        print(processor.encode("🤷"))
        print(processor.decode(processor.encode("hello world!")))
        gpt2_pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        print(processor.find_tokens(gpt2_pattern, "Hello123 world how are you"))
    elif text == "save":
        pass
    elif text == "train":
        model = GPTLanguageModel(processor)
        model.train(True)
    elif text == "exit":
        exit()