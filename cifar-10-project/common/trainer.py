import torch


# 学習して学習結果を返す
def train(model,num_epochs,train_loader,test_loader,optimizer,device,criterion):
    history = []
    for epoch in range(num_epochs):
        n_train_acc,n_eval_acc = 0,0
        train_loss,eval_loss = 0,0
        n_train,n_eval = 0,0
        
        model.train() # 訓練モード発動
        
        for inputs,labels in(train_loader):
            train_batch_size = len(labels)
            n_train += train_batch_size
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            predicted = torch.max(outputs,dim=1)[1]
            n_train_acc += (predicted==labels).sum().item()
            train_loss += loss.item() * train_batch_size
            
        model.eval() # 検証モード発動
        with torch.no_grad():
            for inputs_test,labels_test in(test_loader):
                test_batch_size = len(labels_test)
                n_eval += test_batch_size
                inputs_test = inputs_test.to(device)
                labels_test = labels_test.to(device)
                outputs_test = model(inputs_test)
                loss = criterion(outputs_test,labels_test)
                predicted_test = torch.max(outputs_test,dim=1)[1]
                n_eval_acc += (predicted_test==labels_test).sum().item()
                eval_loss += loss.item() * test_batch_size
        
        # 精度計算
        train_acc = n_train_acc / n_train
        eval_acc = n_eval_acc / n_eval
        avg_train_loss = train_loss / n_train
        avg_eval_loss = eval_loss / n_eval
        print(f'Epoch:{epoch} avg_train_loss:{avg_train_loss} train_acc:{train_acc} avg_eval_loss:{avg_eval_loss} eval_acc:{eval_acc}')
        history.append([epoch,avg_train_loss,train_acc,avg_eval_loss,eval_acc])
    
    return history