import pickle
import neps
from neps.space.neps_spaces.parameters import PipelineSpace


class SimpleSpace(PipelineSpace):
    int_param1 = neps.Integer(1, 100)
    int_param2 = neps.Integer(1, 100)


print("Testing remove()...")
space_remove = SimpleSpace().remove("int_param2")
print(f"  After remove: {list(space_remove.get_attrs().keys())}")
pickled = pickle.dumps(space_remove)
unpickled = pickle.loads(pickled)
print(f"  ✅ Pickle/unpickle successful: {list(unpickled.get_attrs().keys())}")

print("\nTesting add()...")
space_add = SimpleSpace().add(neps.Float(0, 1), "new_float")
print(f"  After add: {list(space_add.get_attrs().keys())}")
pickled = pickle.dumps(space_add)
unpickled = pickle.loads(pickled)
print(f"  ✅ Pickle/unpickle successful: {list(unpickled.get_attrs().keys())}")

print("\nTesting add_prior()...")
space_prior = SimpleSpace().add_prior("int_param1", 50, "medium")
print(f"  After add_prior: {list(space_prior.get_attrs().keys())}")
print(f"  int_param1 has prior: {space_prior.get_attrs()['int_param1'].has_prior}")
pickled = pickle.dumps(space_prior)
unpickled = pickle.loads(pickled)
print(f"  ✅ Pickle/unpickle successful: {list(unpickled.get_attrs().keys())}")
print(
    f"  Unpickled int_param1 has prior: {unpickled.get_attrs()['int_param1'].has_prior}"
)

print("\nAll tests passed! ✅")
