<template>
  <div class="std-block">
    <Dropdown
      v-model="user"
      :options="users"
      filter
      show-clear
      option-label="name"
      class="w-full md:w-14rem select-inner"
    >
    </Dropdown>
    <label>选择被试</label>
    <button @click="selectUsers">选择用户</button>
  </div>
</template>

<script>
import { GetUsername } from '../api/api'
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'

export default {
  setup() {
    const user = ref()
    const users = ref([])
    const router = useRouter()

    const getusername = () => {
      GetUsername().then((res) => {
        users.value = res.data.usernames.map((username) => ({ name: username, selected: false }))
      })
    }

    const goToUserDetail = (username) => {
      ///待更改
      router.push({ name: 'UserDetail', params: { username } })
    }

    const selectUsers = () => {
      const selectedUsers = users.value.filter((user) => user.selected)
      router.push({ name: 'SelectedUsers', query: { users: JSON.stringify(selectedUsers) } })
    }

    onMounted(getusername)

    return {
      users,
      goToUserDetail,
      selectUsers
    }
  }
}
</script>

<style scoped>
.user-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}

.icon {
  cursor: pointer;
}

.select-inner {
  width: 100%;
}

.std-block {
  margin: 2rem 3rem ;
  /* 为每个块设置宽度为容器宽度的一半减去间距 */
  flex: 0 0 calc(50% - 6rem);
  /* 确保块之间有适当的间距 */
  max-width: calc(50% - 6rem);
  /* 使块在flex容器中垂直居中 */
  display: flex;
  align-items: center;
}
</style>
