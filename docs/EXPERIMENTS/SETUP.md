请你了解一下现在vt_franka这一套遥操采集数据和policy inference rollout deployment的整个系统。

我正想着基于现在的数据采集，replay和policy inference，做以下的扩展：

增加gripper_forever_closed的设置：
如果true：
- 在data collection/replay/rollout的每一个episode的开头：按H reset到一个pose之后，如果夹爪是开着的，输出一段提示，等待用户按一个按键确认之后关上gripper，之后的整个episode都是关闭的状态不会再打开gripper，结束一个episode的rollout时也是不需要打开。
如果false：
- 维持现在这样
增加rand_init_pose：
- default是[0, 0, 0]，控制在每一个data collection/replay/rollout的每一个episode的initial pose之上，x y z 加上一个range of randomization（-/+）比如[0.05, 0.05, 0.05]代表initial eef pose 里面每个xzy的维度随机加或者减0.05之内的值。

请你帮我看一下可不可行以及实现的路径。