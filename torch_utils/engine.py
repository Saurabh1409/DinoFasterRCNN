def train(model, train_dataloader, optimizer, scheduler, , epoch, DEVICE, args):
    batch_loss_list = []
    batch_loss_cls_list = []
    batch_loss_box_reg_list = []
    batch_loss_objectness_list = []
    batch_loss_rpn_list = []

    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(train_dataloader) - 1)
    step_counter = 0
    model.train()
    for batch_idx, (images, targets) in tqdm(enumerate(train_dataloader),ascii=True):
        step_counter+=1
        images = torch.stack([img for img in images]).to(DEVICE)
        if args.use_collate:
            targets = [
                {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
                for t in targets
            ]
        else:
            targets = [{k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in targets.items()}]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        scheduler.step()

        batch_loss_list.append(loss_value)
        batch_loss_cls_list.append(loss_dict_reduced['loss_classifier'].detach().cpu())
        batch_loss_box_reg_list.append(loss_dict_reduced['loss_box_reg'].detach().cpu())
        batch_loss_objectness_list.append(loss_dict_reduced['loss_objectness'].detach().cpu())
        batch_loss_rpn_list.append(loss_dict_reduced['loss_rpn_box_reg'].detach().cpu())


        if args.use_wandb and batch_idx%1000==0:
            wandb.log({
                "batch_total_loss":loss_value,
                "RPN_loss":loss_dict_reduced['loss_rpn_box_reg'].detach().cpu(),
                "Classification Loss":loss_dict_reduced['loss_classifier'].detach().cpu(),
                "Bbox_loss":loss_dict_reduced['loss_box_reg'].detach().cpu(),
                "learning_rate": optimizer.param_groups[0]['lr'],
            })

            print(f"-----------Loss after {batch_idx} batches -------------")
            print(f"----------- Total Loss : {loss_value} -------------")

        if scheduler is not None:
            scheduler.step(epoch + (step_counter/len(train_dataloader)))

    return(
        batch_loss_list, 
        batch_loss_cls_list, 
        batch_loss_box_reg_list, 
        batch_loss_objectness_list, 
        batch_loss_rpn_list
    )