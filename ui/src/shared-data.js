// TODO: Replace with Vuex
var state = {
    loggedIn: false,
    loadingState: true,
    userId:"",
    errorLoadingState: false
  }
  
  function setLoggedIn(newValue) {
    state.loggedIn = newValue;
  }

  export default {
    state: state,
    setLoggedIn: setLoggedIn  }