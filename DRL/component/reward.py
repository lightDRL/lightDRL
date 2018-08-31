import scipy.signal
import numpy as np
# ===========================
#   Set rewards
# ===========================
class Reward(object):

    def __init__(self, factor, gamma):
        # Reward parameters
        self.factor = factor
        self.gamma = gamma

    # Set step rewards to total episode reward
    def total(self, ep_batch, tot_reward):
        for step in ep_batch:
            step[2] = tot_reward*self.factor
        return ep_batch


    # Set step rewards to discounted reward
    def discount(self, r_batch):
        # print('START---------in discount-----------')
        # print('factor={}, gamma={}'.format(self.factor, self.gamma))
        x = r_batch

        # print('ep_batch[:,2] -> ' + str(x))
        discounted = scipy.signal.lfilter([1], [1, -self.gamma], x[::-1], axis=0)[::-1]
        # print('afte lfilter ->', discounted)
        discounted *= self.factor
        # print('afte *factor ->', discounted)


        # print('ep_batch[:,2] -> ' + str(ep_batch[i,2]))
        # print('END---------in discount-----------')
        # print('reward discounted = ', discounted)

        return discounted

    # Set step rewards to discounted reward
    def discount_batch(self, ep_batch):
        # print('START---------in discount-----------')
        # print('factor={}, gamma={}'.format(self.factor, self.gamma))
        x = ep_batch[:,2]

        # print('ep_batch[:,2] -> ' + str(x))
        discounted = scipy.signal.lfilter([1], [1, -self.gamma], x[::-1], axis=0)[::-1]
        # print('afte lfilter ->', discounted)
        discounted *= self.factor
        # print('afte *factor ->', discounted)

        for i in range(len(discounted)):
            ep_batch[i,2] = discounted[i]

        # print('ep_batch[:,2] -> ' + str(ep_batch[i,2]))
        # print('END---------in discount-----------')

        return ep_batch

    def discount_ori_print(self, ep_batch):
        # print('START---------in discount-----------')
        print('factor={}, gamma={}'.format(self.factor, self.gamma))
        x = ep_batch[:,2]

        print('ep_batch[:,2] -> ' + str(x))
        discounted = scipy.signal.lfilter([1], [1, -self.gamma], x[::-1], axis=0)[::-1]
        print('afte lfilter ->', discounted)
        discounted *= self.factor
        print('afte *factor ->', discounted)

        for i in range(len(discounted)):
            ep_batch[i,2] = discounted[i]

        print('ep_batch[:,2] -> ' + str(ep_batch[i,2]))
        print('END---------in discount-----------')

        return ep_batch


    def discount_005(self, ep_batch):
        # print('START---------in discount-----------')
        # print('factor={}, gamma={}, len(x)={}'.format(self.factor, self.gamma,len(x)))
        x = ep_batch[:,2]

        # print('ep_batch[:,2] -> ' + str(x))
        discounted_ep_rs = np.zeros_like(x)

        for t in reversed(range(0, len(x))):
            discounted_ep_rs[t] = 0.05        # all 0.05

        # print('discounted_ep_rs -> ' + str(discounted_ep_rs))
        # discounted_ep_rs *= self.factor
        # print('discounted_ep_rs after * factor-> ' + str(discounted_ep_rs))
        for i in range(len(discounted_ep_rs)):
            ep_batch[i,2] = discounted_ep_rs[i]

        # print('ep_batch[:,2] -> ' + str(ep_batch[:,2]))
        # print('END---------in discount-----------')

        return ep_batch


    def discount_add_005(self, ep_batch):
        # print('START---------in discount-----------')
        # print('factor={}, gamma={}, len(x)={}'.format(self.factor, self.gamma,len(x)))
        x = ep_batch[:,2]

        # print('ep_batch[:,2] -> ' + str(x))
        discounted_ep_rs = np.zeros_like(x)
        running_add = 0
        for t in reversed(range(0, len(x))):
            discounted_ep_rs[t] = running_add * 0.05 * 1.0
            running_add+=1

        # print('discounted_ep_rs -> ' + str(discounted_ep_rs))
        # discounted_ep_rs *= self.factor
        # print('discounted_ep_rs after * factor-> ' + str(discounted_ep_rs))
        for i in range(len(discounted_ep_rs)):
            ep_batch[i,2] = discounted_ep_rs[i]

        # print('ep_batch[:,2] -> ' + str(ep_batch[:,2]))
        # print('END---------in discount-----------')

        return ep_batch

    


    def reverse_add_rewards(self, ep_rs, r_dicount = 0.9):
        print('reverse_and_norm_rewards ep_rs -> len = {}, {}'.format(len(ep_rs), ep_rs))
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(ep_rs)
        running_add = 0
        for t in reversed(range(0, len(ep_rs))):
            running_add = running_add * r_dicount + ep_rs[t]
            discounted_ep_rs[t] = running_add
            
        print('reverse_add_rewards -> discounted_ep_rs = ' + str(discounted_ep_rs))
        return discounted_ep_rs