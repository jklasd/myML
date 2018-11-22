import seaborn as sns
import matplotlib.pyplot as plt

batches = get_batches(int_text[:-(batch_size+1)], batch_size, seq_length)
test_batches = get_batches(int_text[-(batch_size+1):], batch_size, seq_length)
top_k = 10
#预测结果的Top K准确率
topk_acc_list = []
topk_acc = 0
#与预测结果距离最近的Top K准确率
sim_topk_acc_list = []
sim_topk_acc = 0

#表示k值是一个范围，不像Top K是最开始的K个
range_k = 5
#以每次训练得出的距离中位数为中心，以范围K为半径的准确率，使用预测结果向量
floating_median_idx = 0
floating_median_acc_range_k = 0
floating_median_acc_range_k_list = []

#同上，使用的是相似度向量
floating_median_sim_idx = 0
floating_median_sim_acc_range_k = 0
floating_median_sim_acc_range_k_list = []

#保存训练损失和测试损失
losses = {'train':[], 'test':[]}
#保存各类准确率
accuracies = {'accuracy':[], 'topk':[], 'sim_topk':[], 'floating_median_acc_range_k':[], 'floating_median_sim_acc_range_k':[]}

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    for epoch_i in range(num_epochs):
        state = sess.run(initial_state, {input_text: batches[0][0]})

        #训练的迭代，保存训练损失
        for batch_i, (x, y) in enumerate(batches):
            feed = {
                input_text: x,
                targets: y,
                initial_state: state,
                lr: learning_rate}
            train_loss, state, _ = sess.run([total_loss, final_state, train_op], feed)  #cost
            losses['train'].append(train_loss)

            # Show every <show_every_n_batches> batches
            if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    epoch_i,
                    batch_i,
                    len(batches),
                    train_loss))

        #使用测试数据的迭代
        acc_list = []
        prev_state = sess.run(initial_state, {input_text: np.array([[1]])})
        for batch_i, (x, y) in enumerate(test_batches):
            # Get Prediction
            test_loss, acc, probabilities, prev_state = sess.run(
                [total_loss, accuracy, probs, final_state],
                {input_text: x,
                 targets: y,
                 initial_state: prev_state})

            #保存测试损失和准确率
            acc_list.append(acc)
            losses['test'].append(test_loss)
            accuracies['accuracy'].append(acc)

            print('Epoch {:>3} Batch {:>4}/{}   test_loss = {:.3f}'.format(
                epoch_i,
                batch_i,
                len(test_batches),
                test_loss))

            #利用嵌入矩阵和生成的预测计算得到相似度矩阵sim
            valid_embedding = tf.nn.embedding_lookup(normalized_embedding, np.squeeze(probabilities.argmax(2)))
            similarity = tf.matmul(valid_embedding, tf.transpose(normalized_embedding))
            sim = similarity.eval()

            #保存预测结果的Top K准确率和与预测结果距离最近的Top K准确率
            topk_acc = 0
            sim_topk_acc = 0
            for ii in range(len(probabilities)):

                nearest = (-sim[ii, :]).argsort()[0:top_k]
                if y[ii] in nearest:
                    sim_topk_acc += 1

                if y[ii] in (-probabilities[ii]).argsort()[0][0:top_k]:
                    topk_acc += 1

            topk_acc = topk_acc / len(y)
            topk_acc_list.append(topk_acc)
            accuracies['topk'].append(topk_acc)

            sim_topk_acc = sim_topk_acc / len(y)
            sim_topk_acc_list.append(sim_topk_acc)
            accuracies['sim_topk'].append(sim_topk_acc)

            #计算真实值在预测值中的距离数据
            realInSim_distance_list = []
            realInPredict_distance_list = []
            for ii in range(len(probabilities)):
                sim_nearest = (-sim[ii, :]).argsort()
                idx = list(sim_nearest).index(y[ii])
                realInSim_distance_list.append(idx)

                nearest = (-probabilities[ii]).argsort()[0]
                idx = list(nearest).index(y[ii])
                realInPredict_distance_list.append(idx)

            print('真实值在预测值中的距离数据：')
            print('max distance : {}'.format(max(realInPredict_distance_list)))
            print('min distance : {}'.format(min(realInPredict_distance_list)))
            print('平均距离 : {}'.format(np.mean(realInPredict_distance_list)))
            print('距离中位数 : {}'.format(np.median(realInPredict_distance_list)))
            print('距离标准差 : {}'.format(np.std(realInPredict_distance_list)))

            print('真实值在预测值相似向量中的距离数据：')
            print('max distance : {}'.format(max(realInSim_distance_list)))
            print('min distance : {}'.format(min(realInSim_distance_list)))
            print('平均距离 : {}'.format(np.mean(realInSim_distance_list)))
            print('距离中位数 : {}'.format(np.median(realInSim_distance_list)))
            print('距离标准差 : {}'.format(np.std(realInSim_distance_list)))

            #计算以距离中位数为中心，范围K为半径的准确率
            floating_median_sim_idx = int(np.median(realInSim_distance_list))
            floating_median_sim_acc_range_k = 0

            floating_median_idx = int(np.median(realInPredict_distance_list))
            floating_median_acc_range_k = 0
            for ii in range(len(probabilities)):
                nearest_floating_median = (-probabilities[ii]).argsort()[0][floating_median_idx - range_k:floating_median_idx + range_k]
                if y[ii] in nearest_floating_median:
                    floating_median_acc_range_k += 1

                nearest_floating_median_sim = (-sim[ii, :]).argsort()[floating_median_sim_idx - range_k:floating_median_sim_idx + range_k]
                if y[ii] in nearest_floating_median_sim:
                    floating_median_sim_acc_range_k += 1

            floating_median_acc_range_k = floating_median_acc_range_k / len(y)
            floating_median_acc_range_k_list.append(floating_median_acc_range_k)
            accuracies['floating_median_acc_range_k'].append(floating_median_acc_range_k)

            floating_median_sim_acc_range_k = floating_median_sim_acc_range_k / len(y)
            floating_median_sim_acc_range_k_list.append(floating_median_sim_acc_range_k)
            accuracies['floating_median_sim_acc_range_k'].append(floating_median_sim_acc_range_k)

        print('Epoch {:>3} floating median sim range k accuracy {} '.format(epoch_i, np.mean(floating_median_sim_acc_range_k_list)))#:.3f
        print('Epoch {:>3} floating median range k accuracy {} '.format(epoch_i, np.mean(floating_median_acc_range_k_list)))#:.3f
        print('Epoch {:>3} similar top k accuracy {} '.format(epoch_i, np.mean(sim_topk_acc_list)))#:.3f
        print('Epoch {:>3} top k accuracy {} '.format(epoch_i, np.mean(topk_acc_list)))#:.3f
        print('Epoch {:>3} accuracy {} '.format(epoch_i, np.mean(acc_list)))#:.3f

    # Save Model
    saver.save(sess, save_dir)
    print('Model Trained and Saved')
    embed_mat = sess.run(normalized_embedding)