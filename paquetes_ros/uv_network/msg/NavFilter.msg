# A msg to publish pose and twist estimation
# and performance of the Navigation Filter

# The pose in this message should be specified in the coordinate frame given by header.frame_id.
Header header

# Estimate
geometry_msgs/PoseWithCovariance[] pose
geometry_msgs/TwistWithCovariance[] twist
geometry_msgs/Vector3 euler

# Sensor parameters
geometry_msgs/Point accel_bias
geometry_msgs/Point gyro_bias

# Performance
float32[] cov_sv
float32 cov_tr