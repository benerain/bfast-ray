# bfast-ray


[![image](https://img.shields.io/pypi/v/bfast-ray.svg)](https://pypi.python.org/pypi/bfast-ray)
<!-- [![image](https://img.shields.io/conda/vn/conda-forge/bfast-ray.svg)](https://anaconda.org/conda-forge/bfast-ray) -->


**The bfast-ray package provides a highly-efficient parallel implementation for the `Breaks For Additive Season and Trend (BFASTmonitor) proposed by Verbesselt et al. The implementation is based on Ray**

This package is adapted from https://github.com/diku-dk/bfast, with credit to mortvest.


-   Free software: MIT license
-   Documentation: https://benerain.github.io/bfast-ray
    

## Dependencies
============
- numpy==1.25.2
- pandas==2.0.3
- scikit-learn==1.3.0
- scipy==1.11.2
- matplotlib==3.7.2
- wget==3.2
- ray==2.6.1

## Input args 
- start_monitor：Python 标准库中的 datetime类型的数据（用datetime库解析字符串）

- freq： 季节性变化的观测频次。比如，监测变量NDWI是按年度变化的，采样数据是某一天的均值，那么freq就设置为365；采样数据是某一月的均值，那么freq就设置为12

- k ： 谐波级数，傅立叶分解的精度

- hfrac： 用多少observation 来计算时间序列的 均值和方差。  moving window the比例 限制为0.25, 0.5, 1

- trend :  bool值， 是否使用offset值

- level： 监测的明显水平（和 ROC，如果选择）过程，即概率类型 I 错误。       

- verbose： int 中间过程输不输出

- backend: str, 'python' or 'python-ray'

- address: str, e.g.:"ray://xxx.xx.xx.xx:xxxxx"

- ray_remote_args: dic, e.g.: {"resources": {"customResources": 1}, "batch_size":10}

## Output 

breaks： 数组。 -2代表 没有充分的历史数据。 -1代表该pixel没有break 。所有其他非负数据对应于第一个在监控期间检测到的中断的索引序号

means： 每一个MOSUM 过程的mean值（例如考虑NDMI指数时，像素的正平均值对应于植被的增加，）

timers : dict 是个字典，包含拟合过程不同阶段的运行时测量值。

共三个

## use example

```py
k = 3
freq = 365
trend = False
hfrac = 0.25
level = 0.05
start_monitor = datetime(2010, 1, 1)

model = BFASTMonitor(
            start_monitor,
            freq=freq,
            k=k,
            hfrac=hfrac,
            trend=trend,
            level=level,
            # backend='python',
            backend='python-ray',
            cluster_address = "ray://xx.xx.xx.xx:xxxxx",
            ray_remote_args= {"resources": {"xxxx": xx}, "batch_size":1}
        )


model.fit(data, dates, nan_value=-32768)

breaks = model.breaks
means = model.means
timers = model.timers
```




