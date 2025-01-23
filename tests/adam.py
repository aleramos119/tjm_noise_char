#%%
import tensorflow as tf


#%%
# Step 1: Define the variables to optimize
x = tf.Variable(0.0, trainable=True)
y = tf.Variable(0.0, trainable=True)

# Step 2: Define the custom function
def custom_function(x, y):
    return (x - 3)**2 + (y + 4)**2

# Step 3: Set up the Adam optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)


#%%
# Step 4: Optimization loop
for step in range(1000):
    # Compute the function value
    loss = custom_function(x, y)
    
    # Step 5: Manually compute gradients
    grad_x = 2 * (x - 3)  # Partial derivative w.r.t. x
    grad_y = 2 * (y + 4)  # Partial derivative w.r.t. y


    
    # Combine gradients into a list
    gradients = [tf.convert_to_tensor(grad_x), tf.convert_to_tensor(grad_y)]
    
    # Step 6: Apply gradients manually using the optimizer
    optimizer.apply_gradients(zip(gradients, [x, y]))
    
    # Print progress every 10 steps
    if step % 10 == 0:
        print(f"Step {step}, Loss: {loss.numpy()}, x: {x.numpy()}, y: {y.numpy()}")

# Final optimized values
print(f"Optimized x: {x.numpy()}, y: {y.numpy()}")




# %%

# %%
