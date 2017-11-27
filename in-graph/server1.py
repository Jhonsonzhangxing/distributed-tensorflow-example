import tensorflow as tf

# def cluster
cluster = tf.train.ClusterSpec({"worker": ["hades02:5222", "hades02:5223"], 
                                "ps" : ["hades02:6222"] })

# creat  Worker server
server = tf.train.Server(cluster,job_name="worker",task_index=0)
server.join()
