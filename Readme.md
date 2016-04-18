### Python class as loss layer in Caffe.
##### An example showing how to use Python class as loss Layer in Caffe.
##### Code mainly comes from [https://gist.github.com//shelhamer/8d9a94cf75e6fb2df221#file-pyloss-py-L28](https://gist.github.com//shelhamer/8d9a94cf75e6fb2df221#file-pyloss-py-L28)


##### Conclusion of callback sequence:

###### initialization:

1. =>  setup()
2. =>  reshape()

###### training:
1. =>  reshape()
2. =>  forward()
3. =>  backward()
4. 

###### notice:
###### remember to put *pyloss.py* in *$PYTHONPATH*, or just copy it to *$CAFFE_HOME/python* for convenience(make sure you had added it to *$PYTHONPATH*).
