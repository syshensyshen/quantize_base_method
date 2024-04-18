import pytest

class Test_Class7():
    @pytest.fixture()
    def test_case1(self):
        uname = ["user1", "user2"]
        return uname

    def test_case2(self, test_case1):
        b = test_case1
        print(b)


if __name__ == '__main__':
    pytest.main()