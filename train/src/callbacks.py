def on_train_batch_end(trainer):
    with open("train_batches.txt", "w") as file:
        file.write("train batch end")
