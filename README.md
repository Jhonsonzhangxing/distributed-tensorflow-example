# distributed-tensorflow-example
an simple example of distributed tensorflow

run in-graph example (just modify hostname before running) 

'''
python server1.py  # run on compute node
python server2.py  # run on compute node
python ps_server1.py  # run on compute node
python mnist_cnn_in_graph.py  # run on client node
'''

run between-graph example

'''
python mnist_cnn_between_graph.py \
--ps_hosts=[hostname1]:[port1],[hostname2]:[port2]...... \
--worker_hosts=[hostname3]:[port3],[hostname4]:[port4]...... \
--job_name=<worker or ps>
--task_index=<index of the program doing same job> 
'''
