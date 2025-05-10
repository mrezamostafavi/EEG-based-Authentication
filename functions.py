def train_one_epoch(model, train_loader, loss_fn, optimizer, epoch=None):
  model.train()
  loss_train = AverageMeter()
  acc_train = Accuracy(task="multiclass", num_classes=9).to(device)
  with tqdm(train_loader, unit="batch") as tepoch:
    for inputs, targets in tepoch:
      if epoch is not None:
        tepoch.set_description(f"Epoch {epoch}")
      inputs = inputs.to(device)
      targets = targets.to(device)

      outputs = model(inputs)


      loss = loss_fn(outputs, targets)

      loss.backward(retain_graph=True)

      optimizer.step()
      optimizer.zero_grad()

      loss_train.update(loss.item())
      acc_train(outputs, targets.int())
      tepoch.set_postfix(loss=loss_train.avg,
                         accuracy=100.*acc_train.compute().item())
  return model, loss_train.avg, acc_train.compute().item()

def train_one_epoch_kd(student, teacher, train_loader, loss_fn, optimizer, epoch=None):
  student.train()
  loss_train = AverageMeter()
  acc_train = Accuracy(task="multiclass", num_classes=9).to(device)
  with tqdm(train_loader, unit="batch") as tepoch:
    for inputs, targets in tepoch:
      if epoch is not None:
        tepoch.set_description(f"Epoch {epoch}")
      inputs = inputs.to(device)
      targets = targets.to(device)

      outputs = student(inputs)


      with torch.no_grad():
        teacher_outputs = teacher(inputs)

      loss = loss_fn_kd(outputs, targets, teacher_outputs, T=10, alpha=0.6)

      loss.backward()

      optimizer.step()
      optimizer.zero_grad()

      loss_train.update(loss.item())
      acc_train(outputs, targets.int())
      tepoch.set_postfix(loss=loss_train.avg,
                         accuracy=100.*acc_train.compute().item())
  return student, loss_train.avg, acc_train.compute().item()

def validation(model, test_loader, loss_fn):
  model.eval()
  with torch.no_grad():
    loss_valid = AverageMeter()
    acc_valid = Accuracy(task="multiclass", num_classes=9).to(device)

    all_targets = []
    all_outputs = []

    for i, (inputs, targets) in enumerate(test_loader):
      inputs = inputs.to(device)
      targets = targets.to(device)

      outputs = model(inputs)
      loss = loss_fn(outputs, targets)

      loss_valid.update(loss.item())
      acc_valid(outputs, targets.int())
      outputs = torch.argmax(outputs, dim=1)

      all_targets.append(targets)
      all_outputs.append(outputs)

    all_targets = torch.cat(all_targets, dim=0)
    all_outputs = torch.cat(all_outputs, dim=0)

  return loss_valid.avg, acc_valid.compute().item(), all_targets, all_outputs