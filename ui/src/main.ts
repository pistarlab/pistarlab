import '@babel/polyfill'
import 'mutationobserver-shim'
import Vue from 'vue'
import './plugins/bootstrap-vue'
import App from './App.vue'
import './registerServiceWorker'
import router from './router'
import store from './store'
import VueSidebarMenu from 'vue-sidebar-menu'
import 'vue-sidebar-menu/dist/vue-sidebar-menu.css'
import "nprogress"

import { library } from '@fortawesome/fontawesome-svg-core'
import { faCoffee } from '@fortawesome/free-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/vue-fontawesome'
import { BootstrapVue, IconsPlugin } from 'bootstrap-vue'


import VueApollo from 'vue-apollo'
import { ApolloClient, DefaultOptions } from 'apollo-client'
import { createHttpLink } from 'apollo-link-http'
import { InMemoryCache } from 'apollo-cache-inmemory'

import {
    appConfig
} from "./app.config"

const httpLink = createHttpLink({
  // You should use an absolute URL here
  uri: appConfig.API_URL+'/graphql',
})

// Cache implementation
const cache = new InMemoryCache()

const defaultOptions:DefaultOptions = {
  watchQuery: {
    fetchPolicy: 'cache-and-network',
    errorPolicy: 'ignore',
  },
  query: {
    fetchPolicy: 'network-only',
    errorPolicy: 'all',
  },
  mutate: {
    errorPolicy: 'all',
  },
};

// Create the apollo client
const apolloClient = new ApolloClient({
  link: httpLink,
  cache,
  defaultOptions
})

const apolloProvider = new VueApollo({
  defaultClient: apolloClient,
})
Vue.use(VueApollo)

library.add(faCoffee)
Vue.use(VueSidebarMenu)
Vue.use(BootstrapVue)
Vue.component('font-awesome-icon', FontAwesomeIcon)

// This imports the dropdown and table plugins
import { DropdownPlugin, TablePlugin } from 'bootstrap-vue'

Vue.use(DropdownPlugin)

Vue.config.productionTip = false



new Vue({
  router,
  store,

  data: {

  },
  apolloProvider,
  render: h => h(App)
}).$mount('#app')  
  