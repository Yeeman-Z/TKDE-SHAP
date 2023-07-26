from matplotlib import pyplot as plt


def mnist_checkup():

    def get_data_for_digit(source, digit):
        output_sequence = []
        # NUM_EXAMPLES_PER_USER = 5000
        all_samples = [i for i, d in enumerate(source[1]) if d == digit]
        for i in range(0, min(len(all_samples)-BATCH_SIZE, NUM_EXAMPLES_PER_USER), BATCH_SIZE):
            batch_samples = all_samples[i:i + BATCH_SIZE]
            output_sequence.append({
                'x':
                    np.array([source[0][i].flatten() / 255.0 for i in batch_samples],
                            dtype=np.float32),
                'y':
                    np.array([source[1][i] for i in batch_samples], dtype=np.int32)
            })
        return output_sequence

    mnist_train, mnist_test = tf.keras.datasets.mnist.load_data()

    federated_train_data = [get_data_for_digit(mnist_train, d) for d in range(10)]

    federated_test_data = [get_data_for_digit(mnist_test, d) for d in range(10)]
    # check the loaded dataset of federated learning
    print("================Check the loaded data=====================")
    
    print("TrainBatchData[0..%d], TrainBatchData[-1][0..%d]"%(len(trainBatchData), len(trainBatchData[-1])))
    print("If we want get dataset of client [1,3,4]:")
    
    subset = [1,3,4]
    subsetTrainBacthData = [trainBatchData[client] for client in subset]
    print("subsetTrainBatchData[0..%d], subsetTrainBatchData[-1][0..%d]"%(len(subsetTrainBacthData), len(subsetTrainBacthData[-1])))


    print("Data Size of each Client:", [len(x)*BATCH_SIZE for x in trainBatchData])
    # print("length of subsetData", len(subsetTrainBacthData))


    # print(trainBatchData[-1][-1]['x'][-1], trainBatchData[-1][-1]['y'][-1])

    # print()
    # plt.imshow(trainBatchData[clientNumber-2][-1]['x'][-1].reshape(28, 28), cmap='gray')
    # plt.grid(False)
    # plt.show()
    # print("show the figure.")
    # plt.savefig("./show_image.png")

    print("================Check the batch_loss=====================")
    initial_model = collections.OrderedDict(
        weights=np.zeros([DATSETSHAPE, DATSETLABEL], dtype=np.float32),
        bias=np.zeros([DATSETLABEL], dtype=np.float32)
    )
    sample_batch = trainBatchData[4][-1]
    print(batch_loss(initial_model, sample_batch))

    print("================Check the batch_train=====================")
    model = initial_model
    losses = []
    for _ in range(5):
        model = batch_train(model, sample_batch, 0.1)
        losses.append(batch_loss(model, sample_batch))
    print("Losses:", losses)

    print("================Check the batch_train=====================")
    locally_trained_model = local_train(initial_model, 0.1, trainBatchData[4])
    print('initial_model loss =', local_eval(initial_model, trainBatchData[4]))
    print('locally_trained_model loss =', local_eval(locally_trained_model, trainBatchData[4]))


    print("===============Check the combination=======================")
    
    itemset = [i for i in range(CLIENT_NUM)]
    print("allset is :", itemset)
    allsubsets = buildPowerSets(itemset)
    print("its subsets are:", allsubsets)

    subset2power = [power2number(subset) for subset in allsubsets]
    print("the subset and its number", subset2power)

    # print("==============Check the data")
    # print()
