import numpy as np
from collections import Counter
import pickle
from copy import deepcopy
import os
import sys

class PolyaUrn:
    def __init__(self, rho=3, nu=2, eta=1., init_colors=5, init_balls_per_color=1, compact=False, shuffle=False, entropy_base=2):

        # Colors initially contained in the urn
        self.init_colors = init_colors
        
        # Amount of balls per color intially contained in the urn
        self.init_balls_per_color = init_balls_per_color
        
        # Rho
        self.rho = rho
        
        # Nu + 1
        self.nu_plus_1 = nu + 1

        # Eta
        self.eta = eta

        self.entropy_base = entropy_base # only 10 or 2
        if entropy_base != 10 and entropy_base != 2:
            raise ValueError('Entropy base must be 2 or 10')

        self.compact = compact
        self.shuffle = shuffle

        self.reset()


    def reset(self):
        # Present amount of balls per color in the urn
        self.balls_per_color = [0] * self.init_colors
        
        # Distribution of the colors IN THE URN (internal colors). It is balls_per_color normalized!
        self.distribution = np.zeros(self.init_colors)

        # Total amount of balls IN THE URN. It must always be equal to the sum of balls_per_color elements
        self.num_balls = self.init_colors * self.init_balls_per_color
        
        # Set of colored balls IN THE URN
        self.balls = []
        for c in range(self.init_colors):
            for k in range(self.init_balls_per_color):
                self.balls.append(c)
                
        # Position of colors in the balls set
        self.ind_colors = dict()
        for k, b in enumerate(self.balls):
            if b in self.ind_colors.keys():
                self.ind_colors[b].append(k)
            else:
                self.ind_colors[b] = [k]

        # Total amount of colors IN THE URN
        self.num_colors = self.init_colors

        # Present set of already sampled colors
        self.seen = set()

        # Initialization of distribution of already seen colors (no colors at the beginning)
        self.seen_distribution = np.array([])

        # Present amount of already seen balls per color
        self.seen_balls_per_color = [0] * self.init_colors

        # Present amount of already seen balls
        self.num_seen_balls = 0

        # Present amount of already seen colors
        self.num_seen_colors = 0

        # Initialization of balls_per_color (IN THE URN)
        for k in range(self.init_colors):
            self.balls_per_color[k] = self.init_balls_per_color

        # Initialization of distribution of colors IN THE URN
        self.distribution = np.array(self.balls_per_color) / self.num_balls

        # Output sequence initialization
        #self.seq = np.array([], dtype=np.int)
        self.seq = []

        # Output sequence of novelties initialization
        #self.seq_novelty = np.array([], dtype=np.bool)
        self.seq_novelty = []
        
        ### Semantic Section #################################
        # Present amount of labels
        self.num_labels = int(np.ceil(self.num_balls / self.nu_plus_1))
        
        # Present amount of balls per label
        self.balls_per_label = [int(np.ceil(self.num_balls / self.num_labels))] * self.num_labels
        if sum(self.balls_per_label) > self.num_balls:
            self.balls_per_label[-1] -= sum(self.balls_per_label) - self.num_balls
            
        # Set of labelled balls IN THE URN
        self.labels = []
        for l in range(self.num_labels):
            self.labels += [l] * self.balls_per_label[l]
        
        # Initialization of distribution of labels IN THE URN
        self.label_distribution = np.array(self.balls_per_label) / self.num_balls
        
        #print(self.balls, self.labels)
        #print(self.label_distribution)
        
        # Initialization of label dicts
        self.label2color = dict()
        self.color2label = dict()
        for l in range(self.num_labels):
            self.label2color[l] = []
        for c in range(self.num_colors):
            label = int(c/int(np.ceil(self.num_balls / self.num_labels)))
            self.color2label[c] = label
            self.label2color[label].append(c)
        
        self.color2offspring = dict()
        for c in range(self.num_colors):
            self.color2offspring[c] = []
        
        self.label2parent_color = dict()
        for l in range(self.num_labels):
            self.label2parent_color[l] = None
        
        self.extras = dict()
        self.extras['seq_entropy'] = []
        self.extras['seq_seen_entropy'] = []
        self.extras['seq_distributions'] = []
        self.extras['seq_seen_distributions'] = []

            
    def sample(self, ext_sample=None):
        if ext_sample is None:
            # Sampling a ball from the urn
            if self.seq == [] or self.eta == 1.0:
                #sample = np.random.choice(np.arange(self.num_colors), p=self.distribution)
                ind = np.random.randint(self.num_balls)
                sample = self.balls[ind]
            else:
                # Semantic Section
                prev_color = self.seq[-1]
                prev_label = self.color2label[prev_color]
                # All colors with the same label as the last element
                colors2reinforce = [prev_color] #self.label2color[prev_label]
                # Offspring of the last element
                colors2reinforce += self.color2offspring[prev_label]
                # Parent of last element label
                lab_pc = self.label2parent_color[prev_label]
                if lab_pc is not None:
                    colors2reinforce += [lab_pc]
                #colors2reinforce_set = set()
                #for c in colors2reinforce:
                #    if c not in colors2reinforce_set:
                #        colors2reinforce_set.add(c)
                colors2reinforce = set(colors2reinforce)
                colors2weaken = set(range(self.num_colors)) - colors2reinforce # - colors2reinforce_set
                #distr = self.distribution.copy()
                #distr[colors2weaken] *= self.eta
                #distr /= distr.sum()
                #sample = np.random.choice(np.arange(self.num_colors), p=distr)
                ind2reinf = []
                for c in colors2reinforce:
                    ind2reinf += self.ind_colors[c]
                    #ind2reinf += list(np.where(self.balls == c)[0])
                ind2weaken = []
                for c in colors2weaken:
                    ind2weaken += self.ind_colors[c]
                #ind2weaken = list(set(range(self.num_balls)) - set(ind2reinf))
                p_reinf = len(ind2reinf) / self.num_balls
                p_weaken = len(ind2weaken) / self.num_balls
                pp = np.array([p_reinf, self.eta * p_weaken])
                pp /= pp.sum()
                cl = np.random.choice([0,1], p=pp)
                if cl == 0:
                    #ind = np.random.choice(ind2reinf)
                    ind0 = np.random.randint(len(ind2reinf))
                    ind = ind2reinf[ind0]
                else:
                    #ind = np.random.choice(ind2weaken)
                    ind0 = np.random.randint(len(ind2weaken))
                    ind = ind2weaken[ind0]
                try:
                    sample = self.balls[ind]
                except:
                    print("Unexpected error:", sys.exc_info()[0])
                    print(ind)
        else:
            if ext_sample < self.num_colors:
                sample = ext_sample
            else:
                raise ValueError('Invalid external sample')
        
        novelty = False
        # If it's a novelty...
        if sample not in self.seen:
            # Updating seen set
            self.seen.add(sample)
            # Increasing the seen_balls_per_color array by adding nu_plus_1 zero elements
            self.seen_balls_per_color += [0] * self.nu_plus_1
            # Add new colors into the urn
            self.balls_per_color += [1] * self.nu_plus_1
            self.num_balls += self.nu_plus_1
            self.num_colors += self.nu_plus_1
            self.num_seen_colors += 1
            novelty = True
            
            # Semantic section
            new_label = self.num_labels
            new_colors = list(range(self.num_colors - self.nu_plus_1, self.num_colors))
            new_ind = list(range(self.num_balls - self.nu_plus_1, self.num_balls))
            for k, c in enumerate(new_colors):
                self.ind_colors[c] = [new_ind[k]]
            self.labels += [new_label] * self.nu_plus_1
            self.balls += new_colors
            self.balls_per_label += [self.nu_plus_1]

            self.num_labels += 1
            for k in new_colors:
                self.color2label[k] = new_label
                self.color2offspring[k] = []
            #self.label2color[new_label] = new_colors

            self.label2parent_color[new_label] = sample
            self.color2offspring[sample] = new_colors
        
        self.seq.append(sample)
        self.seq_novelty.append(novelty)
        
        self.num_seen_balls += 1
        self.seen_balls_per_color[sample] += 1

        # Reinforcing sampled color
        self.balls_per_color[sample] += self.rho
        self.num_balls += self.rho
        
        self.balls += [sample] * self.rho
        self.labels += [self.color2label[sample]] * self.rho
        self.balls_per_label[self.color2label[sample]] += self.rho
        self.balls_per_color[sample] += self.rho
        new_ind = list(range(self.num_balls - self.rho, self.num_balls))
        self.ind_colors[sample] += new_ind

        self.distribution = np.array(self.balls_per_color) / self.num_balls
        self.seen_distribution = np.array(self.seen_balls_per_color) / self.num_seen_balls
        
        self.label_distribution = np.array(self.balls_per_label) / self.num_balls

        if self.entropy_base == 2:
            self.entropy = (- self.distribution * np.log2(self.distribution)).sum()
        elif self.entropy_base == 10:
            self.entropy = (- self.distribution * np.log(self.distribution)).sum()
        else:
            raise ValueError('Entropy logarithm base must be either 2 or 10')

        self.seen_entropy = 0.0
        for sd in self.seen_distribution:
            if sd > 0:
                if self.entropy_base == 2:
                    self.seen_entropy += (-sd * np.log2(sd))
                elif self.entropy_base == 10:
                    self.seen_entropy += (-sd * np.log(sd))

        return sample, novelty


    def seq_compacting(self):
        sorted_seen = sorted(list(self.seen))
        dd = dict()
        for k, ss in enumerate(sorted_seen):
            dd[ss] = k
        new_seq = self.seq.copy()
        for k, s in enumerate(self.seq):
            new_seq[k] = dd[s]
        return new_seq
    
    
    def seq_color_shuffling(self, max_color=None):
        if max_color is None or max_color < self.num_seen_colors:
            max_color = self.seq.max() + 1
        ind = np.arange(max_color)
        np.random.shuffle(ind)
        dd = dict()
        for k, ii in enumerate(ind):
            dd[k] = ii
        new_seq = []
        for k, s in enumerate(self.seq):
            new_seq.append(dd[s])
        return np.array(new_seq)



    def set_entropy_base(self, base):
        if int(base) == 2 or int(ebase) == 10:
            self.entropy_base = base
        else:
            raise ValueError('Entropy base must be 2 or 10')


        
    def get_sequence(self, seq_len=1000, external_sequence=None, print_every=0, init_polya=True,
                     get_seq_entropies=False, get_seq_distributions=False, get_seq_seen_distributions=False):
        if init_polya:
            self.reset()
        
        self.seq = list(self.seq)
        self.seq_novelty = list(self.seq_novelty)

        if external_sequence is not None:
            external_sequence = list(external_sequence)
            seq_len = len(external_sequence)

        
        for k in range(seq_len):
            #print(k, external_sequence)
            if print_every > 0 and k % print_every == 0:
                print('{}/{}'.format(k, seq_len))
            if external_sequence is None:
                sam, nov = self.sample()
            else:
                sam, nov = self.sample(ext_sample=external_sequence[k])

            #self.seq.append(sam)
            #self.seq_novelty.append(nov)
            if get_seq_entropies:
                self.extras['seq_entropy'].append(self.entropy)
                self.extras['seq_seen_entropy'].append(self.seen_entropy)
            if get_seq_distributions:
                self.extras['seq_distributions'].append(self.distribution)
            if get_seq_seen_distributions:
                self.extras['seq_seen_distributions'].append(self.seen_distribution)

        if self.compact:
            self.compacting()

        if self.shuffle:
            self.shuffling()
            
        self.seq = np.array(self.seq)
        self.seq_novelty = np.array(self.seq_novelty)



    def get_entropy_and_distributions(self, get_seq_entropies=True, get_seq_distributions=True, get_seq_seen_distributions=True):
        if len(self.seq) > 0:
            seq = self.seq.copy()
            self.get_sequence(seq_len=len(self.seq), external_sequence=seq,
                              get_seq_entropies=get_seq_entropies,
                              get_seq_distributions=get_seq_distributions,
                              get_seq_seen_distributions=get_seq_seen_distributions)


            
    def __repr__(self):
        ss = ''
        ss += 'Rho: {}\n'.format(self.rho)
        ss += 'Nu: {}\n'.format(self.nu_plus_1 - 1)
        ss += 'Eta: {}\n'.format(self.eta)
        ss += '# balls: {}\n'.format(self.num_balls)
        ss += '# internal colors: {}\n'.format(self.num_colors)
        ss += '# novelties: {}\n'.format(self.seq_novelty.sum())
        ss += 'Polya sequence length: {}\n'.format(len(self.seq))
        if self.extras['seq_entropy'] == []:
            ss += 'Sequence entropy not present\n'
        else:
            ss += 'Sequence entropy done\n'
        if self.extras['seq_distributions'] == []:
            ss += 'Sequence distributions not present\n'
        else:
            ss += 'Sequence distributions done\n'
        if self.extras['seq_seen_distributions'] == []:
            ss += 'Sequence seen distributions not present\n'
        else:
            ss += 'Sequence seen distributions done\n'
        #ss += 'Balls per color: {}\n'.format(self.balls_per_color)
        #ss += 'Distribution: {}\n'.format(self.distribution)
        #ss += 'Seen: {}\n'.format(self.seen)
        return ss
    
    def heaps(self, seq=None, max_seq_len=None, max_colors=None, max_novelties=None):
        if len(self.seq) == 0:
            if max_seq_len is not None:
                self.get_sequence(seq_len=max_seq_len)
            elif max_colors is not None:
                self.get_max_color_sequence(max_colors=max_colors)
            elif max_novelties is not None:
                self.get_max_novelty_sequence(max_novelty=max_novelties)
            else:
                self.get_sequence()
        colors = set()
        heaps = []
        for c in self.seq:
            colors.add(c)
            heaps.append(len(colors))
        return heaps
    
    def zipf(self, max_seq_len=None, max_colors=None, max_novelties=None):
        if len(self.seq) == 0:
            if max_seq_len is not None:
                self.get_sequence(seq_len=max_seq_len)
            elif max_colors is not None:
                self.get_max_color_sequence(max_colors=max_colors)
            elif max_novelties is not None:
                self.get_max_novelty_sequence(max_novelty=max_novelties)
            else:
                self.get_sequence()
        
        seq_list = list(self.seq)
        zipf = [t[1] for t in Counter(seq_list).most_common()]
        return zipf


    def save(self, name_file='', name_dir='polya_seq'):
        # it does NOT save the extra dict!
        extras_temp = deepcopy(self.extras)
        self.extras = dict()
        self.extras['seq_entropy'] = []
        self.extras['seq_seen_entropy'] = []
        self.extras['seq_distributions'] = []
        self.extras['seq_seen_distributions'] = []
        if name_file == '':
            name_file = 'polya_urn_len_'
            name_file += str(len(self.seq))
            name_file += '_novelties_'
            name_file += str(int(self.seq_novelty.sum()))
            name_file += '_nu' + str(int(self.nu_plus_1 - 1))
            name_file += '_rho' + str(int(self.rho))
            name_file += '_eta' + '{:.2f}'.format(self.eta)
            #name_file += '.pkl'
        ind = 1
        name_file_temp = name_file
        while os.path.isfile('./' + name_dir + '/' + name_file_temp + '.pkl'):
            name_file_temp = name_file + '_' + str(ind)
            ind += 1
        name_file = name_file_temp + '.pkl'
        with open('./' + name_dir + '/' + name_file, 'wb') as f:
            pickle.dump(self, f)
        self.extras = extras_temp

    def save_extras(self, name_file='', name_dir='polya_seq'):
        if name_file == '':
            name_file = './polya_urn_len_'
            name_file += str(len(self.seq))
            name_file += '_novelties_'
            name_file += str(int(self.seq_novelty.sum()))
            name_file += '_nu' + str(int(self.nu_plus_1 - 1))
            name_file += '_rho' + str(int(self.rho))
            name_file += '_eta' + '{:.2f}'.format(self.eta)
            name_file += '_extras'
        ind = 1
        name_file_temp = name_file
        while os.path.isfile('./' + name_dir + '/' + name_file_temp + '.pkl'):
            name_file_temp = name_file + '_' + str(ind)
            ind += 1
        name_file = name_file_temp + '.pkl'
        with open('./' + name_dir + '/' + name_file, 'wb') as f:
            pickle.dump(self.extras, f)

    def load_extras(self, name_file):
        with open(name_file, 'rb') as f:
            self.extras = pickle.load(f)



def load_polya(name_file):
    extras = None
    with open(name_file, 'rb') as f:
        polya_obj = pickle.load(f)
    return polya_obj

