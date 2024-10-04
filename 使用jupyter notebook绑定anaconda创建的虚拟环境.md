## 使用jupyter notebook绑定anaconda创建的虚拟环境



```python
#创建虚拟环境
conda create --name my_env
#安装ipykernel
conda install -c anaconda ipykernel
#ipykernel中添加虚拟环境
python -m ipykernel install --user --name=my_env
# restart jupyter notebook

#to delete
#将虚拟环境my_env删除后仍然可以在jupyter notebook中创建以my_env为内核的ipynb文件
#查看当前的jupyter内核列表
jupyter kernelspec list
#手动删除内核
jupyter kernelspec uninstall my_env
```

