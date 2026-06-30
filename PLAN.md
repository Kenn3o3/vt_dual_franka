对于现在的vt franka pipeline我需要你进行一下的modifications：

在模型rollout或者collect data的每一个episode的开头先做以下的东西：
- 对于每一个任务，设定一个home joint，本来是按R之前要先按H（到一个end effector pose），这里不变，但是加一个：按H之前要先按一个J来reset Home Joint到一个预设的值。
- 在按R之前网页stream wrist camera的视角（用来帮助user在了解policy的initial image observation input是什么的前提下设定好场景）
