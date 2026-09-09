"""M048 MVP Demo 子包.

公开 API:
    MVPDemo - 360 相机空间智能最小闭环 Demo

用法:
    from modules.demo import MVPDemo
    demo = MVPDemo()
    demo.run()           # 完整流程
    demo.print_results()  # 打印结果
"""

from __future__ import annotations

from .mvp_demo import MVPDemo

__all__ = ["MVPDemo"]
