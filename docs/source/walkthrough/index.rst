.. _developer-walkthrough-zh:

Code Walkthrough（外部 HOMIE 包）
============================================================

这套文档描述的是重构后的 ``mjlab-homierl``：哪些东西仍然属于上游
``mjlab``，哪些东西在这个仓库里，以及改 HOMIE 任务时应该优先看哪里。

你会在这里学到什么
------------------

- **包边界**：上游 ``mjlab`` 和本仓库各负责什么。
- **代码导航**：HOMIE env cfg、资产、MDP term、HIMPPO runner 在哪里。
- **任务解剖**：H1 HOMIE 任务如何组装，`with_hands` 和 play override
  又分别做了什么。

阅读顺序建议
------------

第一次阅读这次重构后的代码，可以按下面顺序看。这里刻意只保留和本包
直接相关的章节，不再重复整套上游 ``mjlab`` 框架文档。

.. toctree::
   :maxdepth: 2
   :caption: 章节目录

   overview
   project_layout
   tasks_homie_h1
