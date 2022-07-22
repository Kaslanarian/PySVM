import setuptools  #enables develop

setuptools.setup(
    name='pysvm',
    version='0.2',
    description='PySVM : A NumPy implementation of SVM based on SMO algorithm',
    author="Welt",
    author_email="xingcy@smail.nju.edu.cn",
    maintainer="Welt",
    maintainer_email="xingcy@smail.nju.edu.cn",
    packages=['pysvm', 'pysvm/svm'],
    license='MIT License',
    long_description_content_type="text/markdown",
    long_description=open('README.md', encoding='utf-8').read(),
    install_requires=[  #自动安装依赖
        'numpy', 'sklearn'
    ],
    url='https://github.com/Kaslanarian/PySVM',
)
