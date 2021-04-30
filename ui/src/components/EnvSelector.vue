<template>
<div>
    <b-container fluid>
        <div v-for="(spec,idx) in envSpecs" v-bind:key="idx">
            <b-row>
                <b-col>
                    <div>
                        <router-link :to="`/env_spec/view/${spec.ident}`"> {{ spec.displayedName }}</router-link>
                    </div>
                </b-col>
                <b-col>
                    <div>
                        <b-button size="sm" variant="primary" @click="select(spec.ident)">Select</b-button>
                    </div>
                </b-col>
            </b-row>
            <div class="mt-1"></div>
            <b-row>
                <b-col class="">
                    <div>
                        <span class="data_label mt-1">Environment: </span>
                        <span>{{spec.environment.ident}}</span>
                    </div>
                    <div>
                        <span class="data_label mt-1">Version: </span>
                        <span>{{spec.environment.version}}</span>
                    </div>
                    <div>
                        <span class="data_label mt-1">Plugin: </span>
                        <span>{{spec.environment.pluginId}}: {{spec.environment.pluginVersion}}</span>
                    </div>
                </b-col>
                <b-col class="">
                    <div>
                        <span class="data_label mt-1">Type: </span>
                        <span>{{spec.envType}}</span>
                    </div>
                    <div>
                        <span class="data_label mt-1">Description: </span>
                        <span>{{spec.description}}</span>
                    </div>
                </b-col>
            </b-row>
            <hr />
        </div>
    </b-container>
    <div class="mt-2"></div>
</div>
</template>

<script>
import axios from "axios";
import {
    appConfig
} from "../app.config";
import {
    timedelta,
    timepretty
} from "../funcs";
import gql from "graphql-tag";

const GET_ENV_SPECS = gql `
  query {
    envSpecs {
      id
      ident
      displayedName
      config
      meta
      
      envType
      description
      environment{
          id
          ident
          pluginId
            version
            pluginVersion
      }
    }
  }
`;

export default {
    props: {
        //
    },
    apollo: {
        envSpecs: GET_ENV_SPECS,

    },
    data() {
        return {
            envSpecs: [],
            selectedExistingAgent: null

        };
    },
    mounted() {
        //
    },
    methods: {
        select(uid) {
            this.$emit('click', uid)

        },

    },
    computed: {
        //

    },
    // Fetches posts when the component is created.
    created() {
        //

    },
    beforeDestroy() {
        //

    }
};
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->

<style scoped lang="scss">

</style>
