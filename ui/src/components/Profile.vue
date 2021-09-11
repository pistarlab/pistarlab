<template lang="html">
<div class="text-center">

    <br />
    <span v-if="!shared.loggedIn || user_info.last_auth_state!='success'">
        Not signed in, running in <h4>local mode</h4>
        <br/>
        <b-button variant="info" :href="login_uri">Sign-up/Login to piSTAR.ai</b-button>
    </span>
    <span v-else>
        Signed in to piSTAR.ai as <h4>{{user_info.user_id}}</h4>
        <b-button-nav>
        <b-button variant="danger" :href="logout_uri">Logout</b-button>
        </b-button-nav>
    </span>
    <!-- <b-form-input class="mr-3" v-model="agentName" placeholder="(Optional) Enter an agent name" style="width:250px;" ></b-form-input>  -->

</div>
</template>

<script>
//USING https://github.com/chairuosen/vue2-ace-editor
import axios from "axios";
import {
    appConfig
} from "../app.config";
import {
    timedelta,
    timepretty
} from "../funcs";
import gql from "graphql-tag";

export default {
    components: {
        //

    },
    apollo: {
        //
    },
    data() {
        return {
            user_info: {},
            login_uri: null,
            logout_uri:null,

        };
    },
    props: {

    },
    computed: {
        //

    },

    methods: {

        getUserInfo() {
            axios
                .get(`${appConfig.API_URL}/api/profile_data`)
                .then((response) => {
                    console.log(response.data)
                    if (response.data){
                        this.login_uri = response.data.login_uri
                        this.logout_uri = response.data.logout_uri
                        this.user_info = response.data.user_info
                    }
                    else{
                        console.log(response.data.message)
                    }

                })
                .catch((error) => {
                    console.log(error)
                });
        }
    },
    watch: {
        ///
    },
    created() {

        this.getUserInfo()
    },
};
</script>

<style scoped>
.ace_editor {
    font-size: 16px;
}
</style>
