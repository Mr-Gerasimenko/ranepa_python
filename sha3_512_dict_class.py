import hashlib

class sha3_512dict:

    __keys: list
    __values: list

    def __init__(self):
        self.__keys = []
        self.__values = []

    def my_hash(self, x):
        hash_object = hashlib.sha3_512()
        data = x.encode('utf-8')
        hash_object.update(data)
        hex_digest = hash_object.hexdigest()
        return hex_digest
        
    def __getitem__(self, key):
        for idx, my_key in enumerate(self.__keys):
            if self.my_hash(my_key) == self.my_hash(key):
                return self.__values[idx]
        raise KeyError(f"Key {key} not found")

    def __setitem__(self, key, value):
        for idx, my_key in enumerate(self.__keys):
            if self.my_hash(my_key) == self.my_hash(key):
                self.__values[idx] = value
                return self
        self.__keys.append(key)
        self.__values.append(value)

    def __len__(self):
        return len(self.__keys)

    def __repr__(self):
        if len(self.__keys) == 0:
            return "{}"
        s = "{"
        for idx, my_key in enumerate(self.__keys[:-1]):
            s += f"{my_key}: {self.__values[idx]}, "
        s += f"{self.__keys[-1]}: {self.__values[-1]}"
        s += "}"
        return s

    def clear(self):
        self.__keys.clear()
        self.__values.clear()
        return None

    def copy(self):
        copy_dict = sha3_512dict()
        copy_dict.__keys = self.__keys.copy()
        copy_dict.__values = self.__values.copy()
        return copy_dict

    def items(self):
        return list(zip(self.__keys, self.__values))

    def keys(self):
        return self.__keys

    def values(self):
        return self.__values

    def get(self, key, default = None):
        for idx, my_key in enumerate(self.__keys):
            if self.my_hash(my_key) == self.my_hash(key):
                return self.__values[idx]
        return default
    
    def pop(self, key, default = None):
        for idx, my_key in enumerate(self.__keys):
            if self.my_hash(my_key) == self.my_hash(key):
                del self.__keys[idx]
                return_value = self.__values[idx]
                del self.__values[idx]
                return return_value
        return default

    def popitem(self):
        return_key = self.__keys[-1]
        return_value = self.__values[-1]
        del self.__keys[-1]
        del self.__values[-1]
        return return_key, return_value

    def setdefault(self, key, default=None):
        for idx, my_key in enumerate(self.__keys):
            if self.my_hash(my_key) == self.my_hash(key):
                return self.__values[idx]
        self.__keys.append(key)
        self.__values.append(default)
        return default

    def fromkeys(seq, seq_values):
        new_dict = sha3_512dict()
        new_dict.__keys = seq
        new_dict.__values = seq_values
        return new_dict
    
    def update(self, other):
        for item in other.items():
            self.setdefault(*item)
            for my_key in self.__keys:
                if self.my_hash(item[0]) == self.my_hash(my_key):
                    self[my_key] = item[1]
        return None


my_dict = sha3_512dict()
my_dict["name"] = "Alice"
print(my_dict["name"])  
print(len(my_dict))  

print(my_dict.get("age", 30))  
print(my_dict.pop("age", 30))  

my_dict["name"] = "Bob"
print(my_dict["name"]) 

e = sha3_512dict()
print(e)  

d_copy = my_dict.copy()
d_copy["name"] = "Charlie"
print(my_dict["name"])  
print(d_copy["name"])

my_dict.setdefault("role", "Admin")
print(my_dict["role"])

key, val = my_dict.popitem()
print(key, val) 

other = sha3_512dict()
other["name"] = "Dave"
my_dict.update(other)
print(my_dict["name"]) 
