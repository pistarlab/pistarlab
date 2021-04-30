import Vue from 'vue'
import Vuex from 'vuex'
import axios from "axios";


Vue.use(Vuex)

export default new Vuex.Store({
  state: {
    sessions: {},
    envGroups: {},
    agents: {},
    logs: {}
  },
  mutations:
  {
    ADD_LOG_ENTRY(state, log) {
      const key = log['context']
      if (!( key in state.logs))
        state.logs[key] = []
      
      state.logs[key].push(log['msg'])
    }
  },
  actions: {
    updateLog(context, log) {
      context.commit('ADD_LOG_ENTRY', log)

    }

  },
  getters: {

    logs: state => (logType:string) => {
      return state.logs[logType]
    }

  },
  modules: {
  }
})
