import setuptools  #enables develop

setuptools.setup(
    name='pysvm',
    version='0.1',
    description='PySVM : A implementation of LIBSVM',
    author_email="191300064@smail.nju.edu.cn",
    packages=['pysvm'],
    license='MIT License',
    long_description=open('README.md').read(),
    install_requires=[  #自动安装依赖
        'numpy', 'sklearn'
    ],
    url='https://github.com/Kaslanarian/PySVM',
)
