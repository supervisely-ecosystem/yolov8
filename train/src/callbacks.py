def on_train_batch_end(trainer):
    print("train batch end...")
    with open("train_batches.txt", "w") as file:
        file.write("train batch end")
