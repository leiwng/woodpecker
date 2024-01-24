# woodpecker

A tool set for data set process.

## karyotype.py

从核型图中解析出染色体信息，包括轮廓信息，轮廓对应的染色体编号等。

### 程序文件
./config/karyotype.ini : 核型图的基本参数配置文件
./utils/chromo_cv_utils.py : 染色体相关图像处理工具类
./utils/utils.py : 核型图相关图像处理工具类
./doc/karyotype_usage_readme.md : 核型图类的使用说明文档
./requirements.txt : 依赖包列表
./karyotype.py : 核型图类
./karyotype_usage.py : 核型图类的使用示例

### 核型图的基本参数配置

1. 参数配置文件
  ./config/karyotype.ini ， 该文件中包含了核型图的基本参数配置，已经对当前华西的核型图进行了配置，可以直接使用。
  参数在karyotype类实例化时被读取，并以全局变量的形式使用。
  缺省使用，不需要在实例化时传入参数。

### 核型图图片文件的全路径

比如： "D:\\Prj\\github\\woodpecker\\test\\test_img\\L2311245727.001.K.TIF"
一个Karyotype实例对应一个核型图。

### 使用

```python
KYT_IMG_FP = "D:\\Prj\\github\\woodpecker\\test\\test_img\\L2311245727.001.K.TIF"
karyotype_chart = Karyotype(KYT_IMG_FP)
karyotype_chart.read_karyotype()

# 遍历获取每个染色体轮廓信息
for cntrs in karyotype_chart.chromo_cntr_dicts_orgby_cy.values():
    for idx, cntr in enumerate(cntrs):
        pass
"""
在执行完成read_karyotype()方法后，Karyotype 实例中 chromo_cntr_dicts_orgby_cy 属性的内容：
1. 每排的每个染色体轮廓都有染色体编号信息
2. 每排的每个染色体碎片轮廓都同其染色体主干轮廓合并了
3. 数据的排列格式是以cy为key的字典,每个cy对应的value是该cy行的所有的染色体轮廓信息，cy为该排染色体标号中心点的y坐标
   类似于: {
    248: [ {chromo_id:'1',chromo_idx:0,...}, {chromo_id:'2',chromo_idx:1,...},... ],
    468: [ {chromo_id:'6',chromo_idx:5,...}, {chromo_id:'7',chromo_idx:6,...},... ],
    629: [ {chromo_id:'11',chromo_idx:10,...}, {chromo_id:'12',chromo_idx:11,...},... ],
    812: [ {chromo_id:'19',chromo_idx:18,...}, {chromo_id:'20',chromo_idx:19,...},...]
   }
"""

# 另一种遍历方式获取每个染色体轮廓信息
for row_idx, cntrs in karyotype_chart.chromo_cntr_dicts.items():
    for cntr_idx, cntr in enumerate(cntrs):
        pass
"""
在执行完成read_karyotype()方法后，Karyotype 实例 chromo_cntr_dicts 属性的内容：
1. 每排的每个染色体轮廓都有染色体编号信息
2. 每排的每个染色体碎片轮廓都同其染色体主干轮廓合并了
3. 数据的排列格式是以list套list的形式，内层list为每排染色体轮廓的dict，外层list为所有排的list，排的顺序是从上到下的顺序
   类似于: [
    [ {chromo_id:'1',chromo_idx:0,...}, {chromo_id:'2',chromo_idx:1,...},... ],
    [ {chromo_id:'6',chromo_idx:5,...}, {chromo_id:'7',chromo_idx:6,...},... ],
    ...
    [ {chromo_id:'19',chromo_idx:18,...}, {chromo_id:'20',chromo_idx:19,...},...]
   ]
"""
```

## adaptive_threshold_research.py

尝试使用不同的自适应阈值算法对核型图进行二值化处理，以便于后续的轮廓提取。
- bin_thresh_otsu
- bin_thresh_triangle
- bin_thresh_adaptive_mean
- bin_thresh_adaptive_gaussian

## bmp_cvt_to_png.py

遍历目录下的bmp文件，将其转换为png文件，并删除掉原来的bmp文件。

## convert2gray.py

将指定目录下的图片文件转换为灰度图，并保存到指定目录下，不删除原图片。

## save_file_from_many_folders_to_one_folder.py

将保存于多个嵌套目录中的文件，复制到一个目录下。
目录分隔符用"="号代替

## img_id_mapping_btw_source_and_labeling_tool.py

用于映射原始图片文件名与标注系统中的图片文件名，以便于后续的数据处理。


## karyotype_chart_to_classifier_and_seg_training_data.py

将核型图转换为分类模型训练数据。

输出数据集组织形式：
```shell
├───karyotype_chart_to_classifier_training_data_240118 # 该目录为输出根目录
│   ├───0 # 该目录为输出的数据集目录，每个目录对应一个类别, 0对应1号染色体
│   │   ├───A2312010001.001.1.0.png # 该目录下的文件为该类别的图片文件，文件名格式为：标注系统案例号.图片号.染色体编号.同源染色体编号.png
│   │   ├───A2312010001.001.1.1.png
│   │   ...
│   │   ├───A2312100001.033.1.0.png
│   │   ├───A2312100001.033.1.1.png
│   │   ...
│   │   ├───A2401170001.007.1.0.png
│   │   └───A2401170001.007.1.1.png
│   ├───1
│   ├───10
│   ...
│   ├───19
│   ├───2
│   ├───20
│   ├───21
│   ├───22
│   │   ├───A2312010029.001.X.0.png # X染色体对应索引号22
│   │   ├───A2312010029.001.X.1.png
│   │   ...
│   │   ├───A2401170013.007.X.0.png
│   │   └───A2401170013.007.X.1.png
│   ├───23
│   │   ├───A2312010011.003.Y.0.png # Y染色体对应索引号23
│   │   ├───A2312010011.003.Y.1.png
│   │   ...
│   │   ├───A2401170015.027.Y.0.png
│   │   └───A2401170015.027.Y.1.png
│   ├───3
│   ...
│   └───9
```

## karyotype_chart_to_seg_training_data.py

将核型图转换为分割模型训练数据。

输出数据集组织形式：
```shell
├───karyotype_chart_to_seg_training_data_240118 # 该目录为输出根目录
│   ├───crop_picture # 掩码图片根目录
│   │   ├───A2312030011.027 # 单根染色体掩码图片目录，目录名为: 标注系统案例号.图片号
│   │   │   ├───A2312030011.027.M.0_15.png # 图片文件名格式为: 标注系统案例号.图片号.M.染色体索引号_轮廓索引号.png
│   │   │   ├───A2312030011.027.M.0_16.png
│   │   │   ├───A2312030011.027.M.1_19.png
│   │   │   ...
│   │   │   └───A2312030011.027.M.22_68.png
│   │   ├───A2312030027.017
│   │   │   ├───A2312030024.027.M.0_14.png
│   │   │   ...
│   │   │   └───A2312030024.027.M.23_57.png
│   │   ...
│   │   └───A2401170001.001
│   │   │   ├───A2401170001.001.M.0_14.png
│   │   │   ...
│   │   │   └───A2401170001.001.M.23_57.png
│   ├───mid_picture # 中期图片目录，图片文件名格式为: 标注系统案例号.图片号.png，同crop_picture目录下单根染色体掩码图片目录一一对应
│   │   ├───A2312030011.027.png # 图片文件名格式为: 标注系统案例号.图片号.png
│   │   ├───A2312030027.017.png
│   │   ...
│   │   └───A2401170001.001.png

```
