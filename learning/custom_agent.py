class critic(tf.keras.Model):
  def __init__(self):
    super().__init__()
    self.d1 = tf.keras.layers.Dense(1024,activation='relu')
    self.v = tf.keras.layers.Dense(1, activation = None)

  def call(self, input_data):
    x = self.d1(input_data)
    v = self.v(x)
    return v
    

class actor(tf.keras.Model):
  def __init__(self):
    super().__init__()
    self.num_actions = 36
    self.d1 = tf.keras.layers.Dense(1024,activation='relu')
    self.d2 = tf.keras.layers.Dense(512, activation='relu')
    self.a = tf.keras.layers.Dense(self.num_actions, activation='softmax')

  def call(self, input_data):
    layer1 = self.d1(input_data)
    layer2 = self.d1(layer1)
    a = self.a(layer2)
    return a

class CustomAgent(RLAgent):
    def __init__(self, gamma = 0.99):
        
        super().__init__(world, id, json_data)
        self.gamma = gamma
        # self.a_opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
        # self.c_opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
        self.a_opt = tf.keras.optimizers.Adam(learning_rate=7e-3)
        self.c_opt = tf.keras.optimizers.Adam(learning_rate=7e-3)
        self.actor = actor()
        self.critic = critic()
        self.clip_pram = 0.2

        self.state_size = 197
        self.num_actions = 36
    
    def _check_action_space(self):
      action_space = self.get_action_space()
      return action_space == ActionSpace.Continuous

    def _decide_action(self,s):
      # logits = self.custom_model.call(np.reshape(s, [-1, self.state_size]))

      # action_choices = np.arange(self.num_actions)
      # normed_probs = np.linalg.norm(logits, axis=0)
      # action_index = np.random.choice(action_choices, 1, p=normed_probs)
      # custom_action = action_index[0]

      # new_standard_deviation = tf.math.reduce_std(logits)
        
      # custom_logp = calc_logp_gaussian(logits,mean_tf=None,std_tf=new_standard_deviation)
      # return custom_action, custom_logp
      pass
    
    def _get_int_output_path(self):
      assert (self.int_output_dir != '')
      file_path = self.int_output_dir + (
          '/agent{:d}_models/agent{:d}_int_model_{:010d}.ckpt').format(self.id, self.id, self.iter)
      return file_path
    
    def _get_output_path(self):
      assert (self.output_dir != '')
      file_path = self.output_dir + '/agent' + str(self.id) + '_model.ckpt'
      return file_path
    
    def _train_step(self):
      pass

    def load_model(self, in_path):
      # with self.sess.as_default(), self.graph.as_default():
      #   self.saver.restore(self.sess, in_path)
      #   self._load_normalizers()
      #   Logger.print2('Model loaded from: ' + in_path)
      # return
      pass
    
    def save_model(self, out_path):
      # with self.sess.as_default(), self.graph.as_default():
      #   try:
      #     save_path = self.saver.save(self.sess, out_path, write_meta_graph=False, write_state=False)
      #     Logger.print2('Model saved to: ' + save_path)
      #   except:
      #     Logger.print2("Failed to save model to: " + save_path)
      # return
      pass

    def act(self,state):
        prob = self.actor(np.array([state]))
        prob = prob.numpy()
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy()[0])
    
    def actor_loss(self, probs, actions, adv, old_probs, closs):
        
        probability = probs      
        entropy = tf.reduce_mean(tf.math.negative(tf.math.multiply(probability,tf.math.log(probability))))
        #print(probability)
        #print(entropy)
        sur1 = []
        sur2 = []
        
        for pb, t, op in zip(probability, adv, old_probs):
                        t =  tf.constant(t)
                        op =  tf.constant(op)
                        #print(f"t{t}")
                        #ratio = tf.math.exp(tf.math.log(pb + 1e-10) - tf.math.log(op + 1e-10))
                        ratio = tf.math.divide(pb,op)
                        #print(f"ratio{ratio}")
                        s1 = tf.math.multiply(ratio,t)
                        #print(f"s1{s1}")
                        s2 =  tf.math.multiply(tf.clip_by_value(ratio, 1.0 - self.clip_pram, 1.0 + self.clip_pram),t)
                        #print(f"s2{s2}")
                        sur1.append(s1)
                        sur2.append(s2)

        sr1 = tf.stack(sur1)
        sr2 = tf.stack(sur2)
        
        #closs = tf.reduce_mean(tf.math.square(td))
        loss = tf.math.negative(tf.reduce_mean(tf.math.minimum(sr1, sr2)) - closs + 0.001 * entropy)
        #print(loss)
        return loss

    def learn(self, states, actions,  adv , old_probs, discnt_rewards):
        discnt_rewards = tf.reshape(discnt_rewards, (len(discnt_rewards),))
        adv = tf.reshape(adv, (len(adv),))

        old_p = old_probs

        old_p = tf.reshape(old_p, (len(old_p),2))
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            p = self.actor(states, training=True)
            v =  self.critic(states,training=True)
            v = tf.reshape(v, (len(v),))
            td = tf.math.subtract(discnt_rewards, v)
            c_loss = 0.5 * kls.mean_squared_error(discnt_rewards, v)
            a_loss = self.actor_loss(p, actions, adv, old_probs, c_loss)
            
        grads1 = tape1.gradient(a_loss, self.actor.trainable_variables)
        grads2 = tape2.gradient(c_loss, self.critic.trainable_variables)
        self.a_opt.apply_gradients(zip(grads1, self.actor.trainable_variables))
        self.c_opt.apply_gradients(zip(grads2, self.critic.trainable_variables))
        return a_loss, c_loss

def test_reward(env):
  total_reward = 0
  state = env.reset()
  done = False
  while not done:
    action = np.argmax(agentoo7.actor(np.array([state])).numpy())
    next_state, reward, done, _ = env.step(action)
    state = next_state
    total_reward += reward

  return total_reward


def preprocess1(states, actions, rewards, done, values, gamma):
    g = 0
    lmbda = 0.95
    returns = []
    for i in reversed(range(len(rewards))):
       delta = rewards[i] + gamma * values[i + 1] * done[i] - values[i]
       g = delta + gamma * lmbda * dones[i] * g
       returns.append(g + values[i])

    returns.reverse()
    adv = np.array(returns, dtype=np.float32) - values[:-1]
    adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)
    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.int32)
    returns = np.array(returns, dtype=np.float32)
    return states, actions, returns, adv    
