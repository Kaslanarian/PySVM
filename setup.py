import setuptools  #enables develop

setuptools.setup(
    name='pysvm',
    version='0.1',
    description='PySVM : A NumPy implementation of SVM based on SMO algorithm',
    author_email="191300064@smail.nju.edu.cn",
    packages=['pysvm'],
    license='MIT License',
    long_description=open('README.md', encoding='utf-8').read(),
    install_requires=[  #自动安装依赖
        'numpy', 'sklearn'
    ],
    url='https://github.com/Kaslanarian/PySVM',
)
